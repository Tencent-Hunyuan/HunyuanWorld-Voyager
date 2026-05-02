import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import argparse
import numpy as np
import torch
import glob
from scipy.spatial.transform import Rotation
import sys
from PIL import Image
import cv2
import json

# Store original working directory and add VGGT to path
original_cwd = os.getcwd()
vggt_dir = os.path.join(original_cwd, 'vggt')
try:
    os.chdir(vggt_dir)
    if vggt_dir not in sys.path:
        sys.path.insert(0, vggt_dir)
    # Import VGGT modules for pose estimation and depth prediction
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
finally:
    os.chdir(original_cwd)


def process_images_with_vggt(info, image_names, model, device):
    original_images, original_width, original_height = info
    # Preprocess images for VGGT model input
    images = load_and_preprocess_images(image_names).to(device)
    
    # Use bfloat16 for newer GPUs, float16 for older ones
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Run inference with automatic mixed precision
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy arrays and remove batch dimension
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    
    # Extract depth map and convert to world coordinates
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    # Store original images and their metadata
    predictions["original_images"] = original_images
    
    # Normalize images to [0, 1] range and resize to match depth map dimensions
    S, H, W = world_points.shape[:3]
    normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)
    
    for i, img in enumerate(original_images):
        resized_img = cv2.resize(img, (W, H))
        normalized_images[i] = resized_img / 255.0
    
    predictions["images"] = normalized_images
    predictions["original_width"] = original_width
    predictions["original_height"] = original_height
    
    return predictions, image_names


def process_images(image_dir, model, device):
    """
    Process images with VGGT model to extract pose, depth, and camera parameters.
    
    Args:
        image_dir (str): Directory containing input images
        model: VGGT model instance
        device: PyTorch device (CPU/GPU)
    
    Returns:
        tuple: (predictions dict, image_names list)
    """
    # Find all image files in the directory
    image_names = glob.glob(os.path.join(image_dir, "*"))
    image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Limit to 400 images to prevent memory issues
    if len(image_names) > 400:
        image_names = image_names[:400]
    
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Store original images and their dimensions
    original_images = []
    original_width = None
    original_height = None
    
    # Get dimensions from the first image
    first_image = Image.open(image_names[0])
    original_width, original_height = first_image.size
    
    # Load all images as numpy arrays
    for img_path in image_names:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))
    
    return process_images_with_vggt((original_images, original_width, original_height), image_names, model, device)


def extrinsic_to_colmap_format(extrinsics):
    """
    Convert extrinsic matrices from VGGT format to COLMAP format.
    
    VGGT uses camera-to-world transformation matrices (R|t),
    while COLMAP uses quaternion + translation format.
    
    Args:
        extrinsics (np.ndarray): Extrinsic matrices in shape (N, 4, 4)
    
    Returns:
        tuple: (quaternions array, translations array)
    """
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []
    
    for i in range(num_cameras):
        # Extract rotation matrix and translation vector
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        
        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
        
        quaternions.append(quat)
        translations.append(t)
    
    return np.array(quaternions), np.array(translations)

def ToR(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q (np.ndarray): Quaternion in [w, x, y, z] format
    
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    return np.eye(3) + 2 * np.array((
        (-q[2] * q[2] - q[3] * q[3],
        q[1] * q[2] - q[3] * q[0],
        q[1] * q[3] + q[2] * q[0]),
        ( q[1] * q[2] + q[3] * q[0],
        -q[1] * q[1] - q[3] * q[3],
        q[2] * q[3] - q[1] * q[0]),
        ( q[1] * q[3] - q[2] * q[0],
        q[2] * q[3] + q[1] * q[0],
        -q[1] * q[1] - q[2] * q[2])))

def main(image_dir, output_dir):    
    """
    Main function to process images with VGGT and save results in COLMAP format.
    
    Args:
        image_dir (str): Directory containing input images
        output_dir (str): Directory to save output files
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained VGGT model
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    
    # Process images to get predictions
    predictions, image_names = process_images(image_dir, model, device)
    
    # Convert extrinsic matrices to COLMAP format
    quaternions, translations = extrinsic_to_colmap_format(predictions["extrinsic"])
    
    save_dict = {}

    # Extract predictions
    depth = predictions["depth"]
    intrinsic = predictions["intrinsic"]
    height, width = predictions["depth"].shape[1:3]
    ori_height, ori_width = predictions["original_height"], predictions["original_width"]
    
    # Calculate scaling factors for intrinsic matrix adjustment
    s_height, s_width = ori_height / height, ori_width / width
    
    # Process each frame and save results
    for i, (image_name, depth, intrinsic, quaternion, translation) \
        in enumerate(zip(image_names, depth, intrinsic, quaternions, translations)):
        # Convert quaternion back to rotation matrix
        qw, qx, qy, qz = quaternion
        rot = ToR(np.array([qw, qx, qy, qz]))
        trans = translation.reshape(3,1)
        
        # Construct world-to-camera transformation matrix
        bottom = np.array([[0, 0, 0, 1]])
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        
        # Scale intrinsic matrix to original image dimensions
        intrinsic[0, :] = intrinsic[0, :] * s_width
        intrinsic[1, :] = intrinsic[1, :] * s_height
        
        # Save depth map as EXR file
        cv2.imwrite(os.path.join(output_dir, 'depth', f"frame_{(i+1):05d}.exr"), depth, \
            [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
        
        # Store metadata for this frame
        save_dict[f"{(i+1):03d}"] = {
            'image_path': image_name,
            'depth_path': os.path.join(output_dir, 'depth', f"frame_{(i+1):05d}.exr"),
            'intrinsic': intrinsic.tolist(),
            'w2c': w2c.tolist()
        }
    
    # Save all metadata to JSON file
    with open(os.path.join(output_dir, "colmap_data.json"), "w") as f:
        json.dump(save_dict, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VGGT data engine.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    args = parser.parse_args()
    main(args.image_dir, args.output_dir)
