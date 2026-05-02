import numpy as np
from PIL import Image
import torch
import argparse
import os
import json
import imageio
import pyexr
import cv2

try:
    from moge.model.v1 import MoGeModel
except:
    from MoGe.moge.model.v1 import MoGeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./example.png")
    parser.add_argument("--render_output_dir", type=str, default="../demo/example/")
    parser.add_argument("--type", type=str, default="forward",
        choices=["forward", "backward", "left", "right", "turn_left", "turn_right"])
    return parser.parse_args()


def camera_list(
    num_frames=49,
    type="forward",
    Width=512,
    Height=512,
    fx=256,
    fy=256
):
    assert type in ["forward", "backward", "left", "right", "turn_left", "turn_right"], "Invalid camera type"

    start_pos = np.array([0, 0, 0])
    end_pos = np.array([0, 0, 0])
    if type == "forward":
        end_pos = np.array([0, 0, 1])
    elif type == "backward":
        end_pos = np.array([0, 0, -1])
    elif type == "left":
        end_pos = np.array([-1, 0, 0])
    elif type == "right":
        end_pos = np.array([1, 0, 0])

    cx = Width // 2
    cy = Height // 2

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    intrinsics = np.stack([intrinsic] * num_frames)

    # Interpolate camera positions along a straight line
    camera_centers = np.linspace(start_pos, end_pos, num_frames)
    target_start = np.array([0, 0, 100])  # Target point
    if type == "turn_left":
        target_end = np.array([-100, 0, 0])
    elif type == "turn_right":
        target_end = np.array([100, 0, 0])
    else:
        target_end = np.array([0, 0, 100])
    target_points = np.linspace(target_start, target_end, num_frames * 2)[:num_frames]

    extrinsics = []
    for t, target_point in zip(camera_centers, target_points):
        if type == "left" or type == "right":
            target_point = t + target_point
        
        z = (target_point - t)
        z = z / np.linalg.norm(z)
        x = np.array([1, 0, 0])
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)

        R = np.stack([x, y, z], axis=0)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ t
        extrinsics.append(w2c)
    extrinsics = np.stack(extrinsics)

    return intrinsics, extrinsics


# from VGGT: https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py
def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points


def render_from_cameras_videos(points, colors, extrinsics, intrinsics, height, width):
    
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    render_list = []
    mask_list = []
    depth_list = []
    # Render from each camera
    for frame_idx in range(len(extrinsics)):
        # Get corresponding camera parameters
        extrinsic = extrinsics[frame_idx]
        intrinsic = intrinsics[frame_idx]
        
        camera_coords = (extrinsic @ homogeneous_points.T).T[:, :3]
        projected = (intrinsic @ camera_coords.T).T
        uv = projected[:, :2] / projected[:, 2].reshape(-1, 1)
        depths = projected[:, 2]    
        
        pixel_coords = np.round(uv).astype(int)  # pixel_coords (h*w, 2)      
        valid_pixels = (  # valid_pixels (h*w, )      valid_pixels is the valid pixels in width and height
            (pixel_coords[:, 0] >= 0) & 
            (pixel_coords[:, 0] < width) & 
            (pixel_coords[:, 1] >= 0) & 
            (pixel_coords[:, 1] < height)
        )
        
        pixel_coords_valid = pixel_coords[valid_pixels]  # (h*w, 2) to (valid_count, 2)
        colors_valid = colors[valid_pixels]
        depths_valid = depths[valid_pixels]
        uv_valid = uv[valid_pixels]
        
        
        valid_mask = (depths_valid > 0) & (depths_valid < 60000) # & normal_angle_mask
        colors_valid = colors_valid[valid_mask]
        depths_valid = depths_valid[valid_mask]
        pixel_coords_valid = pixel_coords_valid[valid_mask]

        # Initialize depth buffer
        depth_buffer = np.full((height, width), np.inf)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized depth buffer update
        if len(pixel_coords_valid) > 0:
            rows = pixel_coords_valid[:, 1]
            cols = pixel_coords_valid[:, 0]
                            
            # Sort by depth (near to far)
            sorted_idx = np.argsort(depths_valid)
            rows = rows[sorted_idx]
            cols = cols[sorted_idx]
            depths_sorted = depths_valid[sorted_idx]
            colors_sorted = colors_valid[sorted_idx]

            # Vectorized depth buffer update
            depth_buffer[rows, cols] = np.minimum(
                depth_buffer[rows, cols], 
                depths_sorted
            )
            
            # Get the minimum depth index for each pixel
            flat_indices = rows * width + cols  # Flatten 2D coordinates to 1D index
            unique_indices, idx = np.unique(flat_indices, return_index=True)
            
            # Recover 2D coordinates from flattened indices
            final_rows = unique_indices // width
            final_cols = unique_indices % width
            
            image[final_rows, final_cols] = colors_sorted[idx, :3].astype(np.uint8)

        mask = np.zeros_like(depth_buffer, dtype=np.uint8)
        mask[depth_buffer != np.inf] = 255
        
        render_list.append(image)
        mask_list.append(mask)
        depth_list.append(depth_buffer)
    
    return render_list, mask_list, depth_list


def create_video_input(
    render_list, mask_list, depth_list, render_output_dir,
    separate=True, ref_image=None, ref_depth=None,
    Width=512, Height=512,
    min_percentile=2, max_percentile=98
):
    video_output_dir = os.path.join(render_output_dir)
    os.makedirs(video_output_dir, exist_ok=True)
    video_input_dir = os.path.join(render_output_dir, "video_input")
    os.makedirs(video_input_dir, exist_ok=True)

    value_list = []
    for i, (render, mask, depth) in enumerate(zip(render_list, mask_list, depth_list)):

        # Sky part is the region where depth_max is, also included in mask
        mask = mask > 0
        # depth_max = np.max(depth)
        # non_sky_mask = (depth != depth_max)
        # mask = mask & non_sky_mask
        depth[mask] = 1 / (depth[mask] + 1e-6)
        depth_values = depth[mask]
        
        min_percentile = np.percentile(depth_values, 2)
        max_percentile = np.percentile(depth_values, 98)
        value_list.append((min_percentile, max_percentile))

        depth[mask] = (depth[mask] - min_percentile) / (max_percentile - min_percentile)
        depth[~mask] = depth[mask].min()
        

        # resize to 512x512
        render = cv2.resize(render, (Width, Height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize((mask.astype(np.float32) * 255).astype(np.uint8), \
            (Width, Height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (Width, Height), interpolation=cv2.INTER_LINEAR)

        # Save mask as png
        mask_path = os.path.join(video_input_dir, f"mask_{i:04d}.png")
        imageio.imwrite(mask_path, mask)
        
        if separate:
            render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)
            depth_path = os.path.join(video_input_dir, f"depth_{i:04d}.exr")
            pyexr.write(depth_path, depth)  
        else:
            render = np.concatenate([render, depth], axis=-3)
            render_path = os.path.join(video_input_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)

        if i == 0:
            if separate:
                ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                imageio.imwrite(ref_image_path, ref_image)
                ref_depth_path = os.path.join(video_output_dir, f"ref_depth.exr")
                pyexr.write(ref_depth_path, depth) 
            else:
                ref_image = np.concatenate([ref_image, depth], axis=-3)
                ref_image_path = os.path.join(video_output_dir, f"ref_image.png")
                imageio.imwrite(ref_image_path, ref_image)

    with open(os.path.join(video_output_dir, f"depth_range.json"), "w") as f:
        json.dump(value_list, f)


if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl", local_files_only=False).to(device)

    image = np.array(Image.open(args.image_path).convert("RGB").resize((1280, 720)))
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
    output = model.infer(image_tensor)
    depth = np.array(output['depth'].detach().cpu())
    depth[np.isinf(depth)] = depth[~np.isinf(depth)].max() + 1e4
    
    Height, Width = image.shape[:2]
    intrinsics, extrinsics = camera_list(
        num_frames=1, type=args.type, Width=Width, Height=Height, fx=256, fy=256
    )

    # Backproject point cloud
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map.reshape(-1, 3)
    colors = image.reshape(-1, 3)
    
    intrinsics, extrinsics = camera_list(
        num_frames=49, type=args.type, Width=Width//2, Height=Height//2, fx=128, fy=128
    )
    render_list, mask_list, depth_list = render_from_cameras_videos(
        points, colors, extrinsics, intrinsics, height=Height//2, width=Width//2
    )
    
    create_video_input(
        render_list, mask_list, depth_list, args.render_output_dir, separate=True, 
        ref_image=image, ref_depth=depth, Width=Width, Height=Height)
