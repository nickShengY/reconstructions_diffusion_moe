import torch
import torch.nn.functional as F
import math
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from models.swin_ae.swin_ae import SwinAutoencoder
from torch.utils.data import DataLoader
from models.diffusion.networks import VPPrecond, EDMPrecond
import cv2
from torchvision import transforms
import torchvision
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import os
from deepface import DeepFace

def add_rayleigh_fading():
    pass

def calculate_batch_psnr(img1, img2):
    psnr_values = []
    batch_size = img1.size(0)
    for i in range(batch_size):
        psnr = cv2.PSNR(img1[i].cpu().numpy(), img2[i].cpu().numpy(), 1.0)
        psnr_values.append(psnr)

    return psnr_values

def load_ckpt_to_model(ckpt_path, model, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"load model state dict with loss: {ckpt['loss']:.4f}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def denormalize_imagenet(tensor):
    """Your existing denormalization function"""
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def afhq_transform():
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def afhq_data_loader(afhq_dataset_path, batch_size, shuffle=True, num_workers=4):
    afhq_dataset = torchvision.datasets.ImageFolder(root=afhq_dataset_path, transform=afhq_transform())
    return DataLoader(afhq_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def add_awgn_noise(images, snr_db):
    dims = list(range(1, images.ndim))  # [1, 2, 3] for images [B, C, H, W]
    signal_power = images.pow(2).mean(dim=dims, keepdim=True)  # [B, 1, 1, 1]
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(images) * torch.sqrt(noise_power)
    noisy_images = images + noise
    return noisy_images

def add_poisson_noise(images, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    lambda_scale = snr_linear ** 2  # SNR = sqrt(lambda), so lambda = SNR^2

    # Compute current mean intensity across batch/channel/spatial dims
    dims = list(range(1, images.ndim))  # [1, 2, 3] for [B, C, H, W]
    current_mean = images.mean(dim=dims, keepdim=True)  # [B, 1, 1, 1]

    # Shift to make non-negative (Poisson requires >= 0)
    shift = torch.clamp(-images.min(), min=0)  # Shift by the most negative value to make min=0
    shifted_images = images + shift

    # Compute mean of shifted images
    shifted_mean = shifted_images.mean(dim=dims, keepdim=True)

    # Avoid division by zero
    scale_factor = torch.where(
        shifted_mean > 0,
        lambda_scale / shifted_mean,
        torch.ones_like(shifted_mean)
    )

    # Scale shifted images
    scaled_shifted = shifted_images * scale_factor

    # Ensure non-negative for Poisson
    scaled_shifted = torch.clamp(scaled_shifted, min=0)

    # Add Poisson noise
    noisy_scaled_shifted = torch.poisson(scaled_shifted)

    # Rescale and shift back
    noisy_shifted = noisy_scaled_shifted / scale_factor
    noisy_images = noisy_shifted - shift

    return noisy_images

def test_load():
    transform = transforms.Compose([
        transforms.Resize((448 , 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    afhq_dataset = torchvision.datasets.ImageFolder(root=r'C:\Users\hungl\OneDrive\Desktop\LatentDiffusionModel\data\afhq_lite', transform=transform)

    img,lbl = afhq_dataset[0]
    img2,lbl2 = afhq_dataset[2001]
    img3,lbl3 = afhq_dataset[3002]
    imgs = torch.stack([img, img2, img3], dim=0)
    lbls = torch.tensor([lbl, lbl2, lbl3])

    imgs = imgs.to("cuda")
    lbls = lbls.to("cuda")
    return imgs, lbls


def load_models(swin_ckpt_path, diffusion_ckpt_path, dm_model_type="EDM"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Load Swin AutoEncoder
    swin_ae = SwinAutoencoder().to(device)
    swin_ckpt = torch.load(swin_ckpt_path, map_location=device)
    swin_ae.load_state_dict(swin_ckpt['model_state_dict'], strict=True)
    swin_ae.eval()
    print(f"Loaded Swin AE with loss: {swin_ckpt['loss']:.4f}")

    # 2. Load Diffusion Model
    if dm_model_type == "VP":
        diffusion_model = VPPrecond(
            img_resolution=16,  # Match your training config
            img_channels=128,   # Match Swin encoder output channels
            label_dim=3,        # No conditional classes
            model_type="SongUNet",
        ).to(device)
    elif dm_model_type == "EDM":
        diffusion_model = EDMPrecond(
            img_resolution=16,  # Match your training config
            img_channels=128,   # Match Swin encoder output channels
            label_dim=3,        # No conditional classes
            model_type="SongUNet",
        ).to(device)

    # Load the latest checkpoint from your training
    diffusion_ckpt = torch.load(diffusion_ckpt_path, map_location=device)
    diffusion_model.load_state_dict(diffusion_ckpt['model_state_dict'], strict=False)
    diffusion_model.eval()

    for param in diffusion_model.parameters():
        param.requires_grad = False

    print(f"Loaded Diffusion Model with loss: {diffusion_ckpt['loss']:.4f}")

    return swin_ae, diffusion_model

def get_clip_seg():
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda").eval()
    return processor, model

def crop_bbox_face(input_img, processor, model, save_dir="crop_bbox", thr=0.3, pad_ratio=0.1):
    device = "cuda"
    pil_img = Image.open(input_img).convert("RGB")
    orig_w, orig_h = pil_img.size

    inputs = processor(text="face", images=pil_img, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model(**inputs)
    mask = torch.sigmoid(output.logits)

    mask_np = mask.detach().cpu().numpy()[0]

    # Resize mask (fix warning: use Image.BILINEAR instead of resample)
    mask_pil = Image.fromarray((mask_np * 255).astype("uint8"))
    mask_resized = mask_pil.resize((orig_w, orig_h), Image.BILINEAR)  # ← Fixed
    mask_arr = np.array(mask_resized).astype(float) / 255.0
    binary = mask_arr > thr

    ys, xs = np.where(binary)
    if xs.size == 0 or ys.size == 0:
        print(f"❌ No face detected in {input_img}")  # ← Debug message
        print(f"   Try lowering threshold (current: {thr})")
        return None

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    pad = int(max(orig_w, orig_h) * pad_ratio)
    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = min(orig_w, xmax + pad)
    ymax = min(orig_h, ymax + pad)

    cropped = pil_img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))

    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(input_img)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(save_dir, f"{name}_crop{ext}")
    cropped.save(out_path)
    print(f"✅ Saved cropped face to: {out_path}")

    return out_path

def face_identification(person_img, database_imgs, thr=0.7, model_name="Facenet512"):
    min_distance = float('inf')
    closest_img = None

    for db_img_path in database_imgs:
        try:
            result = DeepFace.verify(person_img, db_img_path, model_name=model_name, enforce_detection=True)
            distance = result['distance']
            if distance < min_distance:
                min_distance = distance
                closest_img = db_img_path
        except Exception as e:
            print(f"{db_img_path}: Failed - {str(e)[:50]}")
        continue

    if min_distance > thr:
        print(f"No match found. Distance: {min_distance:.4f} exceeds threshold {thr}")
        closest_img = None
    else:
        print(f"Match found: {closest_img} with distance {min_distance:.4f}")

    return min_distance, closest_img

# plot_person_and_closest_match
def plot_two_similar_faces(person_img_path, match_img_path):
    person_img = Image.open(person_img_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(person_img)
    axes[0].set_title("Person Image")
    axes[0].axis('off')

    if match_img_path is None:
        axes[1].text(0.5, 0.5, 'No Match Found', horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1].axis('off')
    else:
        match_img = Image.open(match_img_path).convert("RGB")
        axes[1].imshow(match_img)
        axes[1].set_title("Closest Match")
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def segment_face_mask(input_img, processor, model, save_dir="segmented_faces", thr=0.3, background="transparent"):
    """
    Extract face using instance segmentation (mask-based extraction)

    Args:
        input_img: Path to input image
        processor: CLIPSeg processor
        model: CLIPSeg model
        save_dir: Output directory
        thr: Threshold for binary mask (0-1)
        background: "transparent", "white", or "black"

    Returns:
        Path to saved segmented image with masked face
    """
    device = "cuda"
    pil_img = Image.open(input_img).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Get segmentation mask
    inputs = processor(text="human face", images=pil_img, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        output = model(**inputs)
    mask = torch.sigmoid(output.logits)
    mask_np = mask.detach().cpu().numpy()[0]

    # Resize mask to original image size
    mask_pil = Image.fromarray((mask_np * 255).astype("uint8"))
    mask_resized = mask_pil.resize((orig_w, orig_h), Image.BILINEAR)
    mask_arr = np.array(mask_resized).astype(float) / 255.0

    # Create binary mask
    binary_mask = (mask_arr > thr).astype(np.uint8)

    # Check if face detected
    if binary_mask.sum() == 0:
        print(f"❌ No face detected in {input_img}")
        print(f"   Try lowering threshold (current: {thr})")
        return None

    # Convert PIL image to numpy array
    img_arr = np.array(pil_img)

    # Apply mask to extract face
    if background == "transparent":
        # Create RGBA image with transparency
        img_rgba = np.dstack([img_arr, binary_mask * 255])
        segmented = Image.fromarray(img_rgba.astype(np.uint8), mode='RGBA')
        ext = ".png"  # Must use PNG for transparency
    elif background == "white":
        # White background
        white_bg = np.ones_like(img_arr) * 255
        segmented_arr = img_arr * binary_mask[:, :, np.newaxis] + white_bg * (1 - binary_mask[:, :, np.newaxis])
        segmented = Image.fromarray(segmented_arr.astype(np.uint8), mode='RGB')
        ext = ".png"
    elif background == "black":
        # Black background
        segmented_arr = img_arr * binary_mask[:, :, np.newaxis]
        segmented = Image.fromarray(segmented_arr.astype(np.uint8), mode='RGB')
        ext = ".png"
    else:
        raise ValueError(f"Unknown background type: {background}")

    # Save segmented image
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(input_img)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(save_dir, f"{name}_segmented{ext}")
    segmented.save(out_path)
    print(f"✅ Saved segmented face to: {out_path}")

    return out_path


def segment_and_crop_face(input_img, processor, model, save_dir="segmented_faces", thr=0.3, pad_ratio=0.1, background="white"):
    """
    Segment face and crop to tight bounding box with masked pixels
    (Combines instance segmentation + bbox cropping)

    Args:
        input_img: Path to input image
        processor: CLIPSeg processor
        model: CLIPSeg model
        save_dir: Output directory
        thr: Threshold for binary mask
        pad_ratio: Padding around detected face region
        background: "transparent", "white", or "black"

    Returns:
        Path to saved segmented+cropped image
    """
    device = "cuda"
    pil_img = Image.open(input_img).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Get segmentation mask
    inputs = processor(text="human face", images=pil_img, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        output = model(**inputs)
    mask = torch.sigmoid(output.logits)
    mask_np = mask.detach().cpu().numpy()[0]

    # Resize mask to original size
    mask_pil = Image.fromarray((mask_np * 255).astype("uint8"))
    mask_resized = mask_pil.resize((orig_w, orig_h), Image.BILINEAR)
    mask_arr = np.array(mask_resized).astype(float) / 255.0

    # Create binary mask
    binary_mask = (mask_arr > thr).astype(np.uint8)

    # Find bounding box of mask
    ys, xs = np.where(binary_mask)
    if xs.size == 0 or ys.size == 0:
        print(f"❌ No face detected in {input_img}")
        return None

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    # Add padding
    pad = int(max(orig_w, orig_h) * pad_ratio)
    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = min(orig_w, xmax + pad)
    ymax = min(orig_h, ymax + pad)

    # Crop mask and image to bounding box
    cropped_mask = binary_mask[ymin:ymax, xmin:xmax]
    img_arr = np.array(pil_img)
    cropped_img = img_arr[ymin:ymax, xmin:xmax]

    # Apply mask to cropped region
    if background == "transparent":
        # RGBA with transparency
        img_rgba = np.dstack([cropped_img, cropped_mask * 255])
        segmented = Image.fromarray(img_rgba.astype(np.uint8), mode='RGBA')
        ext = ".png"
    elif background == "white":
        # White background
        white_bg = np.ones_like(cropped_img) * 255
        segmented_arr = cropped_img * cropped_mask[:, :, np.newaxis] + white_bg * (1 - cropped_mask[:, :, np.newaxis])
        segmented = Image.fromarray(segmented_arr.astype(np.uint8), mode='RGB')
        ext = ".png"
    elif background == "black":
        # Black background
        segmented_arr = cropped_img * cropped_mask[:, :, np.newaxis]
        segmented = Image.fromarray(segmented_arr.astype(np.uint8), mode='RGB')
        ext = ".png"
    else:
        raise ValueError(f"Unknown background type: {background}")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.basename(input_img)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(save_dir, f"{name}_seg_crop{ext}")
    segmented.save(out_path)
    print(f"✅ Saved segmented+cropped face to: {out_path}")

    return out_path