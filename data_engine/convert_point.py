import argparse
import os
import json
import numpy as np
import imageio

from create_input import depth_to_world_coords_points, camera_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--frame_id", type=int, default=0)
    parser.add_argument("--max_depth", type=float, default=25)
    return parser.parse_args()


def save_ply(points: np.ndarray, colors: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = points.shape[0]
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{float(p[0])} {float(p[1])} {float(p[2])} {int(c[0])} {int(c[1])} {int(c[2])}\n")


if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder_path
    video_path = args.video_path
    frame_id = args.frame_id
    max_depth = args.max_depth

    reader = imageio.v2.get_reader(video_path)
    for i, frame in enumerate(reader):
        if i == frame_id:
            frame = frame.astype(np.uint8)
            break

    with open(os.path.join(folder_path, "depth_range.json"), "r") as f:
        depth_range = json.load(f)[frame_id]

    rgb = frame[:512]
    depth = frame[512:, :, 0] / 255.0
    depth = depth * (depth_range[1] - depth_range[0]) + depth_range[0]
    depth = 1 / (depth + 1e-6)
    valid_mask = np.logical_and(depth > 0, depth < max_depth)

    intrinsics, extrinsics = camera_list(
        num_frames=1, type="forward", Width=512, Height=512, fx=256, fy=256
    )
    point_map = depth_to_world_coords_points(depth, extrinsics[0], intrinsics[0])
    points = point_map[valid_mask].reshape(-1, 3)
    colors = rgb[valid_mask].reshape(-1, 3)

    out_ply = os.path.join(folder_path, f"frame_{frame_id:06d}.ply")
    save_ply(points, colors, out_ply)
    print(f"Saved point cloud: {out_ply}, number of points: {points.shape[0]}")
