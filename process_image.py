import os
import cv2
import hydra
import torch
import argparse
import scipy
import skimage
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from accelerate import load_checkpoint_and_dispatch
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from model import fetch_model
from util.util import convert_filepath, generate_disparity, warp_image
from util.padder import InputPadder


def inference_mono(model, image, scale, mean, std):
    """Mono depth estimation inference"""
    h, w = image.shape[-2:]
    if scale > 1:
        image = F.interpolate(image, size=(h // scale, w // scale), mode='bicubic', align_corners=True).clip(0, 255)

    image = (image / 255. - mean) / std
    padder = InputPadder(image.shape, divis_by=14)
    image = padder.pad(image)[0]
    flipped_image = torch.flip(image, [-1])

    with torch.no_grad():
        inverse_depth = model.forward(image)
        torch.cuda.empty_cache()
        flipped_inverse_depth = torch.flip(model.forward(flipped_image), [-1])
        torch.cuda.empty_cache()        

    inverse_depth = padder.unpad(inverse_depth[:, None])
    flipped_inverse_depth = padder.unpad(flipped_inverse_depth[:, None])

    if scale > 1:
        inverse_depth = F.interpolate(inverse_depth, size=(h, w), mode='bilinear', align_corners=True)
        flipped_inverse_depth = F.interpolate(flipped_inverse_depth, size=(h, w), mode='bilinear', align_corners=True)        

    return inverse_depth, flipped_inverse_depth


def inference_stereo(cfg, accelerator, model, warped_image, mask_nocc, mask_inpaint, scale, h, w):
    """Stereo generation inference"""
    warped_image = warped_image / 127.5 - 1
    masked_image = warped_image * (mask_inpaint < 0.5)

    padder = InputPadder(masked_image.shape, divis_by=8)
    masked_image, mask_inpaint = padder.pad(masked_image, mask_inpaint)

    with accelerator.autocast():
        image_pred = model.single_infer(masked_image, mask_inpaint, cfg.num_inference_step)

        image_pred = padder.unpad(image_pred)
        mask_inpaint = padder.unpad(mask_inpaint)
        image_pred = image_pred * (mask_inpaint >= 0.5) + warped_image * (mask_inpaint < 0.5)

    if scale > 1:
        image_pred = F.interpolate(image_pred, size=(h, w), mode='bicubic', align_corners=True).clip(-1, 1)
        mask_nocc = F.interpolate(mask_nocc, size=(h, w), mode='nearest')
        mask_inpaint = F.interpolate(mask_inpaint, size=(h, w), mode='nearest')

    return image_pred, mask_nocc, mask_inpaint


def process_mono_depth(cfg_mono, accelerator_mono, model_mono, image_path, output_dir, dilate_iteration=1):
    """Process mono image to generate depth/disparity"""
    logger = get_logger(__name__)
    
    # Load and prepare image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    image = image.to(accelerator_mono.device)
    
    # Setup output paths
    image_name = Path(image_path).stem
    conf_file = os.path.join(output_dir, f'{image_name}_confidence.npy')
    disp_file = os.path.join(output_dir, f'{image_name}_disparity.npy')
    
    # Run inference with automatic scale adjustment for OOM
    mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(accelerator_mono.device)
    std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(accelerator_mono.device)
    
    scale = 1
    while True:
        try:
            inverse_depth, flipped_inverse_depth = inference_mono(model_mono, image, scale, mean, std)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            scale += 1
            logger.info(f'OOM encountered, retrying with scale {scale}')

    # Calculate confidence and disparity
    normalized_inverse_depth = (inverse_depth - inverse_depth.min()) / (inverse_depth.max() - inverse_depth.min())
    flipped_normalized_inverse_depth = (flipped_inverse_depth - flipped_inverse_depth.min()) / (flipped_inverse_depth.max() - flipped_inverse_depth.min())
    confidence = torch.ones_like(normalized_inverse_depth) - (torch.abs(normalized_inverse_depth - flipped_normalized_inverse_depth))
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())

    disparity = generate_disparity(normalized_inverse_depth).squeeze().cpu().numpy()

    # Post-process disparity
    h, w = disparity.shape[-2:]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disparity = cv2.dilate(disparity, kernel, iterations=dilate_iteration)
    edge = skimage.filters.sobel(disparity) > 3
    disparity[edge] = 0
    disparity = scipy.interpolate.griddata(
        np.stack([ys[~edge].ravel(), xs[~edge].ravel()], 1), 
        disparity[~edge].ravel(), 
        np.stack([ys.ravel(), xs.ravel()], 1), 
        method='nearest'
    ).reshape(h, w)

    # Save outputs
    np.save(conf_file, confidence.squeeze().cpu().numpy())
    np.save(disp_file, disparity)
    
    torch.cuda.empty_cache()
    
    logger.info(f'Saved disparity and confidence for {image_name}')
    return disp_file


def process_stereo_generation(cfg_stereo, accelerator_stereo, model_stereo, image_path, disp_path, output_dir):
    """Process warped image to generate stereo pair"""
    logger = get_logger(__name__)
    
    # Load image and disparity
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32)[..., :3]
    disp = np.load(disp_path)
    
    h, w = image.shape[:2]
    
    # Setup output paths
    image_name = Path(image_path).stem
    right_file = os.path.join(output_dir, f'{image_name}_right.png')
    mask_nocc_file = os.path.join(output_dir, f'{image_name}_mask_nocc.png')
    mask_inpaint_file = os.path.join(output_dir, f'{image_name}_mask_inpaint.png')
    
    # Calculate initial scale
    scale = min(h, w) // 1000
    if scale < 1:
        scale = 1
        
    # Prepare warped image and masks
    if scale > 1:
        image_scaled = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC).clip(0, 255)
        disp_scaled = cv2.resize(disp, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR) / scale
    else:
        image_scaled = image
        disp_scaled = disp
    
    warped_image, mask_nocc, mask_inpaint = warp_image(image_scaled, disp_scaled)
    
    warped_image = torch.from_numpy(warped_image).permute(2, 0, 1).float().unsqueeze(0)
    mask_nocc = torch.from_numpy(mask_nocc)[None, None].float()
    mask_inpaint = torch.from_numpy(mask_inpaint)[None, None].float()
    
    warped_image = warped_image.to(accelerator_stereo.device)
    mask_nocc = mask_nocc.to(accelerator_stereo.device)
    mask_inpaint = mask_inpaint.to(accelerator_stereo.device)
    
    # Run inference with automatic scale adjustment for OOM
    while True:
        try:
            image_pred, mask_nocc, mask_inpaint = inference_stereo(
                cfg_stereo, accelerator_stereo, model_stereo, 
                warped_image, mask_nocc, mask_inpaint, scale, h, w
            )
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            scale += 1
            logger.info(f'OOM encountered, retrying with scale {scale}')
            
            # Recompute with new scale
            if scale > 1:
                image_scaled = cv2.resize(image, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC).clip(0, 255)
                disp_scaled = cv2.resize(disp, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR) / scale
            else:
                image_scaled = image
                disp_scaled = disp
            
            warped_image, mask_nocc, mask_inpaint = warp_image(image_scaled, disp_scaled)
            
            warped_image = torch.from_numpy(warped_image).permute(2, 0, 1).float().unsqueeze(0)
            mask_nocc = torch.from_numpy(mask_nocc)[None, None].float()
            mask_inpaint = torch.from_numpy(mask_inpaint)[None, None].float()
            
            warped_image = warped_image.to(accelerator_stereo.device)
            mask_nocc = mask_nocc.to(accelerator_stereo.device)
            mask_inpaint = mask_inpaint.to(accelerator_stereo.device)
    
    # Convert to images
    image_pred = ((image_pred + 1.) * 127.5).squeeze().round().permute(1, 2, 0).cpu().numpy().astype('uint8')
    mask_nocc = (mask_nocc * 255.).squeeze().round().cpu().numpy().astype('uint8')
    mask_inpaint = (mask_inpaint * 255.).squeeze().round().cpu().numpy().astype('uint8')

    image_pred = Image.fromarray(image_pred)
    mask_nocc = Image.fromarray(mask_nocc)
    mask_inpaint = Image.fromarray(mask_inpaint)

    # Save outputs
    image_pred.save(right_file, lossless=True)
    mask_nocc.save(mask_nocc_file, lossless=True)
    mask_inpaint.save(mask_inpaint_file, lossless=True)

    torch.cuda.empty_cache()
    
    logger.info(f'Saved stereo pair for {image_name}')


def get_image_files(input_path):
    """Get list of image files from path (single file or directory)"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return [str(input_path)]
        else:
            raise ValueError(f"Input file {input_path} is not a supported image format")
    elif input_path.is_dir():
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(input_path.glob(ext))
        return [str(f) for f in sorted(image_files)]
    else:
        raise ValueError(f"Input path {input_path} does not exist")


@hydra.main(version_base=None, config_path='config', config_name='generate_mono')
def main(cfg: DictConfig) -> None:
    """
    Main processing function for unified mono depth estimation and stereo pair generation.
    
    This script combines the functionality of generate_mono.py and generate_stereo.py
    into a single pipeline that processes individual images or folders of images.
    """
    logger = get_logger(__name__)
    
    # Parse additional command line arguments
    usage_examples = """
Examples:

  1. Process a single image:
     python process_image.py --input /path/to/image.jpg --output ./results

  2. Process a folder of images:
     python process_image.py --input /path/to/image/folder --output ./batch_results

  3. High-quality processing with more inference steps:
     python process_image.py --input /path/to/image.jpg --output ./high_quality \\
         --num_inference_steps 100 --dilate_iteration 2

  4. Using a specific GPU:
     CUDA_VISIBLE_DEVICES=0 python process_image.py --input /path/to/image.jpg --output ./results

  5. Custom checkpoint path:
     python process_image.py --input /path/to/image.jpg --output ./results \\
         --mono_checkpoint /path/to/custom/checkpoint.pth

Output files for each input image 'name.jpg':
  - name_disparity.npy      : Disparity map
  - name_confidence.npy     : Confidence map
  - name_right.png          : Generated right stereo image
  - name_mask_nocc.png      : Non-occlusion mask
  - name_mask_inpaint.png   : Inpainting mask
"""
    
    parser = argparse.ArgumentParser(
        description='ZeroStereo: Unified mono depth estimation and stereo pair generation',
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True, help='Input image file or directory')
    parser.add_argument('--output', type=str, default='output', help='Output directory (default: output)')
    parser.add_argument('--mono_checkpoint', type=str, default=None, help='Path to mono depth checkpoint (default: from config)')
    parser.add_argument('--stereo_config', type=str, default='config/generate_stereo.yaml', help='Path to stereo config (default: config/generate_stereo.yaml)')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps for stereo generation (default: 50)')
    parser.add_argument('--dilate_iteration', type=int, default=1, help='Number of dilation iterations for disparity (default: 1)')
    args, unknown = parser.parse_known_args()
    
    # Get image files
    image_files = get_image_files(args.input)
    if not image_files:
        logger.error("No image files found!")
        return
    
    logger.info(f'Found {len(image_files)} image(s) to process')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================
    # Step 1: Setup mono depth model
    # ============================
    logger.info('=' * 50)
    logger.info('Step 1: Loading mono depth estimation model')
    logger.info('=' * 50)
    
    accelerator_mono = instantiate(cfg.accelerator)
    model_mono = fetch_model(cfg, logger)
    
    checkpoint_path = args.mono_checkpoint if args.mono_checkpoint else cfg.checkpoint
    model_mono = load_checkpoint_and_dispatch(model_mono, checkpoint_path)
    model_mono.eval()
    logger.info(f'Loaded mono checkpoint from {checkpoint_path}')
    
    model_mono = accelerator_mono.prepare(model_mono)
    set_seed(cfg.seed, device_specific=True)
    
    # ============================
    # Step 2: Process mono depth for all images
    # ============================
    logger.info('=' * 50)
    logger.info('Step 2: Generating depth/disparity maps')
    logger.info('=' * 50)
    
    disp_files = []
    for image_file in tqdm(image_files, desc='Processing depth', disable=not accelerator_mono.is_main_process):
        disp_file = process_mono_depth(
            cfg, accelerator_mono, model_mono, 
            image_file, str(output_dir), 
            dilate_iteration=args.dilate_iteration
        )
        disp_files.append(disp_file)
    
    # Clean up mono model
    del model_mono
    torch.cuda.empty_cache()
    accelerator_mono.free_memory()
    
    # ============================
    # Step 3: Setup stereo generation model
    # ============================
    logger.info('=' * 50)
    logger.info('Step 3: Loading stereo generation model')
    logger.info('=' * 50)
    
    # Load stereo config
    from hydra import compose, initialize_config_dir
    config_dir = os.path.join(os.getcwd(), 'config')
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg_stereo = compose(config_name='generate_stereo')
    
    # Override num_inference_steps if provided
    cfg_stereo.num_inference_step = args.num_inference_steps
    
    accelerator_stereo = instantiate(cfg_stereo.accelerator)
    model_stereo = fetch_model(cfg_stereo, logger)
    model_stereo.to(accelerator_stereo.device, torch.float16 if cfg_stereo.accelerator.mixed_precision == 'fp16' else torch.float32)
    model_stereo.unet.eval()
    
    set_seed(cfg_stereo.seed, device_specific=True)
    
    # ============================
    # Step 4: Generate stereo pairs for all images
    # ============================
    logger.info('=' * 50)
    logger.info('Step 4: Generating stereo pairs')
    logger.info('=' * 50)
    
    for image_file, disp_file in tqdm(
        zip(image_files, disp_files), 
        total=len(image_files),
        desc='Processing stereo', 
        disable=not accelerator_stereo.is_main_process
    ):
        process_stereo_generation(
            cfg_stereo, accelerator_stereo, model_stereo,
            image_file, disp_file, str(output_dir)
        )
    
    # Clean up stereo model
    accelerator_stereo.free_memory()
    
    logger.info('=' * 50)
    logger.info(f'Processing complete! Results saved to {output_dir}')
    logger.info('=' * 50)
    logger.info('Generated files for each image:')
    logger.info('  - <name>_disparity.npy: Disparity map')
    logger.info('  - <name>_confidence.npy: Confidence map')
    logger.info('  - <name>_right.png: Generated right stereo image')
    logger.info('  - <name>_mask_nocc.png: Non-occlusion mask')
    logger.info('  - <name>_mask_inpaint.png: Inpainting mask')


if __name__ == '__main__':
    main()

