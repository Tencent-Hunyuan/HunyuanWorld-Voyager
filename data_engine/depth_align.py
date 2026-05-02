import os, re
import json
import numpy as np
import cv2
import torch
import imageio
import pyexr
import trimesh
from PIL import Image  

from create_input import render_from_cameras_videos


class DepthAlignMetric:
    """
    深度缩放与相机参数更新处理器
    
    Attributes:
        moge_depth_dir (str): MOGe待处理深度目录
        vggt_depth_dir (str): VGGT待处理深度目录
        vggt_camera_json_file (str): VGGT关联的JSON文件目录
        output_root (str): 输出根目录
    """
    
    def __init__(self, 
                 input_rgb_dir: str,
                 moge_depth_dir: str, 
                 vggt_depth_dir: str, 
                 metric3d_depth_dir: str,
                 vggt_camera_json_file: str, 
                 output_root: str):
        """        
        Args:
            moge_depth_dir: MOGe原始深度路径
            vggt_depth_dir: VGGT原始深度路径
            vggt_camera_json_file: VGGT关联JSON路径
            output_root: 输出根目录，默认为./processed
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # align depth and camera pose to metric level
        self.moge_depth_dir = moge_depth_dir
        self.vggt_depth_dir = vggt_depth_dir
        self.metric3d_depth_dir = metric3d_depth_dir
        self.vggt_camera_json_file = vggt_camera_json_file
        self.output_root = output_root
        
        # depth to pointmap
        self.metric_intrinsic = None
        self.metric_w2c = None
        self.input_rgb_dir = input_rgb_dir 
        self.input_color_paths = []

        
        # output depth / camera pose / pointmap
        self.output_metric_depth_dir = os.path.join(output_root, "output_metric_depth_dir")
        self.output_metric_camera_json = os.path.join(output_root, "output_metric_camera_json")
        self.output_metric_pointmap_dir = os.path.join(output_root, "output_metric_pointmap_dir")
        os.makedirs(self.output_metric_depth_dir, exist_ok=True)
        os.makedirs(self.output_metric_camera_json, exist_ok=True)
        os.makedirs(self.output_metric_pointmap_dir, exist_ok=True)

    def align_depth_scale(self):
        # align Moge depth to VGGT
        moge_align_depth_list, valid_mask_list = self.scale_moge_depth()
        
        # align moge depth and camera pose to metric depth
        self.align_metric_depth(moge_align_depth_list, valid_mask_list)
    
    
    
    def segment_sky_with_oneformer(self, image_path, skyseg_processor, skyseg_model, SKY_CLASS_ID, save_path=None):
        from PIL import Image
        image = Image.open(image_path)
        inputs = skyseg_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(skyseg_model.device)
        
        with torch.no_grad():
            outputs = skyseg_model(**inputs)
        
        # 获取语义分割结果
        predicted_semantic_map = skyseg_processor.post_process_semantic_segmentation(outputs, \
            target_sizes=[image.size[::-1]])[0]
                
        # 提取天空区域
        sky_mask = (predicted_semantic_map == SKY_CLASS_ID).cpu().numpy().astype(np.uint8) * 255
        
        # erosion sky
        kernel = np.ones((3,3), np.uint8)  
        sky_mask = cv2.erode(sky_mask, kernel, iterations=1)  
        
        # 如果需要保存
        if save_path:
            cv2.imwrite(save_path, sky_mask)
        
        return sky_mask

    def get_valid_depth(self, vggt_files, moge_files, input_rgb_files, skyseg_processor, skyseg_model, SKY_CLASS_ID):
        moge_align_depth_list = []
        valid_mask_list = []
        all_valid_max_list = []
        
        for vggt_file, moge_file, input_rgb_file in zip(vggt_files, moge_files, input_rgb_files):
            # 读取深度数据           
            depth_moge = pyexr.read(os.path.join(self.moge_depth_dir, moge_file)).squeeze()
            depth_vggt = pyexr.read(os.path.join(self.vggt_depth_dir, vggt_file)).squeeze()
            depth_vggt = cv2.resize(depth_vggt,  dsize=(depth_moge.shape[1], depth_moge.shape[0]), \
                interpolation=cv2.INTER_LINEAR)

            depth_vggt = torch.from_numpy(depth_vggt).float().to(self.device)
            depth_moge = torch.from_numpy(depth_moge).float().to(self.device)
            
            
            # segmentation sky
            sky_ima_path = os.path.join(self.input_rgb_dir, input_rgb_file)
            sky_mask = self.segment_sky_with_oneformer(sky_ima_path, skyseg_processor, skyseg_model, SKY_CLASS_ID)    
            sky_mask_tensor = torch.from_numpy(sky_mask).float().to(self.device)
            sky_mask = (sky_mask_tensor > 0)  # 天空区域为True
            
            valid_masks = (                              # (H, W)
                torch.isfinite(depth_moge) &               
                (depth_moge > 0) &                      
                torch.isfinite(depth_vggt) &              
                (depth_vggt > 0)     &
                ~sky_mask                 # 非天空区域
            )
                        
            # depth_moge 无效部分 设置为 有效部分最大值的1.5倍   避免final_align_depth出现负数
            depth_moge[~valid_masks] = depth_moge[valid_masks].max() * 1
                                                        
            source_inv_depth = 1.0 / depth_moge
            target_inv_depth = 1.0 / depth_vggt
            
            # print(f'倒数值:{source_inv_depth.min()}, {source_inv_depth.max()}')    # 0.03 ～ 2.2

            source_mask, target_mask = valid_masks, valid_masks
                 
            # Remove outliers  2/8分最合适
            outlier_quantiles = torch.tensor([0.2, 0.8], device=self.device)

            source_data_low, source_data_high = torch.quantile(
                source_inv_depth[source_mask], outlier_quantiles
            )
            target_data_low, target_data_high = torch.quantile(
                target_inv_depth[target_mask], outlier_quantiles
            )
            source_mask = (source_inv_depth > source_data_low) & (
                source_inv_depth < source_data_high
            )
            target_mask = (target_inv_depth > target_data_low) & (
                target_inv_depth < target_data_high
            )
            
            
            mask = torch.logical_and(source_mask, target_mask)
            mask = torch.logical_and(mask, valid_masks)

            source_data = source_inv_depth[mask].view(-1, 1)
            target_data = target_inv_depth[mask].view(-1, 1)

            ones = torch.ones((source_data.shape[0], 1), device=self.device)
            source_data_h = torch.cat([source_data, ones], dim=1)
            transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

            scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
            aligned_inv_depth = source_inv_depth * scale + bias
            
            
            valid_inv_depth = aligned_inv_depth > 0  # 创建新的有效掩码
            valid_masks = valid_masks & valid_inv_depth  # 合并到原有效掩码
            valid_mask_list.append(valid_masks)
                        
            final_align_depth = 1.0 / aligned_inv_depth
            moge_align_depth_list.append(final_align_depth)
            
            all_valid_max_list.append(final_align_depth[valid_masks].max().item())

        return moge_align_depth_list, valid_mask_list, all_valid_max_list


    def scale_moge_depth(self):
        vggt_files = sorted(f for f in os.listdir(self.vggt_depth_dir) if f.endswith('.exr'))
        moge_files = sorted(f for f in os.listdir(self.moge_depth_dir) if f.endswith('.exr'))
        input_rgb_files = sorted(f for f in os.listdir(self.input_rgb_dir) if f.endswith('.png'))
        
        if len(vggt_files) != len(moge_files):
            raise ValueError("文件数量不匹配")
        
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        skyseg_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
        skyseg_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")
        skyseg_model.to(self.device)
        # 定义天空类别的ID 119
        SKY_CLASS_ID = 119
        
        moge_align_depth_list, valid_mask_list, all_valid_max_list = self.get_valid_depth(
            vggt_files, moge_files, input_rgb_files, skyseg_processor, skyseg_model, SKY_CLASS_ID
        )

        # 计算所有帧的有效最大值的中位数
        valid_max_array = np.array(all_valid_max_list)
        q50 = np.quantile(valid_max_array, 0.50)  # 计算50%分位点
        filtered_max = valid_max_array[valid_max_array <= q50]  # 过滤超过分位点的异常值
        
        # 取过滤后数据的最大值（正常范围内的最大值）
        global_avg_max = np.max(filtered_max)
        max_sky_value = global_avg_max * 5
        max_sky_value = np.minimum(max_sky_value, 1000)    # 相对深度最远不能超过 1000
                
        # 统一设置所有帧的无效区域值
        for i, (moge_depth, valid_mask) in enumerate(zip(moge_align_depth_list, valid_mask_list)):
            moge_depth[~valid_mask] = max_sky_value
            
            # 统计超限点占比（在clamp之前）
            over_count = torch.sum(moge_depth > max_sky_value).item()
            total_pixels = moge_depth.numel()
            over_ratio = over_count / total_pixels * 100
            
            
            moge_depth = torch.clamp(moge_depth, max=max_sky_value)  
            moge_align_depth_list[i] = moge_depth  # 更新处理后的深度图
            
        return moge_align_depth_list, valid_mask_list
        
     

    def align_metric_depth(self, moge_align_depth_list, valid_mask_list):
        # 获取metric文件列表
        metric_files = sorted(f for f in os.listdir(self.metric3d_depth_dir) if f.endswith('.exr'))
        
        metric_scales_list = []
        # 遍历所有深度图对
        for idx, (metric_file, moge_depth) in enumerate(zip(metric_files, moge_align_depth_list)):
            
            depth_metric3d = pyexr.read(os.path.join(self.metric3d_depth_dir, metric_file)).squeeze()
            depth_metric3d = torch.from_numpy(depth_metric3d).float().to(self.device)
            
            # 获取对应帧的掩码
            valid_mask = valid_mask_list[idx].to(self.device)
            
            # 提取有效区域数据
            valid_metric = depth_metric3d[valid_mask]
            valid_moge = moge_depth[valid_mask]
            
            # 分位数差计算
            metric_diff = torch.quantile(valid_metric, 0.8) - torch.quantile(valid_metric, 0.2)
            moge_diff = torch.quantile(valid_moge, 0.8) - torch.quantile(valid_moge, 0.2)
            metric_scale = metric_diff / moge_diff
            metric_scales_list.append(metric_scale.cpu().numpy())
            
        # 计算全局平均缩放因子
        metric_scales_mean = np.mean(metric_scales_list)

        # 应用全局缩放 保存 metric depth
        for idx, (metric_file, moge_depth) in enumerate(zip(metric_files, moge_align_depth_list)):
            metric_moge_depth = (moge_depth * metric_scales_mean).cpu().numpy()
            
            # 保存深度文件
            output_path = os.path.join(
                self.output_metric_depth_dir,
                f"{os.path.splitext(metric_file)[0]}_metric.exr"
            )
            pyexr.write(output_path, metric_moge_depth, channel_names=["Y"])

        # 阶段3：更新相机参数
        with open(self.vggt_camera_json_file, 'r') as f:
            camera_data = json.load(f)
        
        # 更新所有帧的平移分量
        for frame_info in camera_data.values():
            w2c_matrix = np.array(frame_info['w2c'])
            w2c_matrix[:3, 3] *= metric_scales_mean  # 直接使用计算好的全局缩放因子
            frame_info['w2c'] = w2c_matrix.tolist()
        
        # 保存更新后的相机参数
        output_json_path = os.path.join(
            self.output_metric_camera_json,
            os.path.basename(self.vggt_camera_json_file)
        )
        with open(output_json_path, 'w') as f:
            json.dump(camera_data, f, indent=4)
    
    
    def load_metirc_camera_parameters(self):  # 修改：增加color_dir参数
        metric_camera_json = os.path.join(self.output_metric_camera_json, 'colmap_data.json')
        with open(metric_camera_json, 'r') as f:
            data = json.load(f)
        
        # load metric camera parameters
        sorted_frames = sorted(data.items(), key=lambda x: int(x[0]))
        first_frame_key, first_frame_data = sorted_frames[0]
        self.metric_intrinsic = [np.array(frame['intrinsic']) for frame in data.values()]
        self.metric_w2c = [np.array(frame['w2c']) for frame in data.values()]
        
        # 加载pointmap input rgb 文件路径
        self.input_color_paths = sorted(
            [os.path.join(self.input_rgb_dir, f) for f in os.listdir(self.input_rgb_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
    
    
    
    def depth_to_pointmap(self):
        
        num_frames = len(self.metric_w2c)
        for frame_index in range(num_frames):
            
            exr_path = os.path.join(self.output_metric_depth_dir, f"frame_{frame_index+1:05d}_metric.exr")
            depth_data = pyexr.read(exr_path).squeeze()
            depth_tensor = torch.from_numpy(depth_data).to(self.device, torch.float32)

            
            # 生成点云
            height, width = depth_tensor.shape
            K_tensor = torch.from_numpy(self.metric_intrinsic[frame_index]).to(device=self.device, dtype=torch.float32)
            w2c = torch.from_numpy(self.metric_w2c[frame_index]).to(device=self.device, dtype=torch.float32)
    
            camtoworld = torch.inverse(w2c)  
            
            # 生成相机坐标系坐标
            u = torch.arange(width, device=self.device).float()
            v = torch.arange(height, device=self.device).float()
            u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
            
            fx, fy = K_tensor[0, 0], K_tensor[1, 1]
            cx, cy = K_tensor[0, 2], K_tensor[1, 2]
            
            x_cam = (u_grid - cx) * depth_tensor / fx
            y_cam = (v_grid - cy) * depth_tensor / fy
            z_cam = depth_tensor
            
            cam_coords_points = torch.stack([x_cam, y_cam, z_cam], dim=-1)
                            
            R_cam_to_world = camtoworld[:3, :3]
            t_cam_to_world = camtoworld[:3, 3]
            world_coords_points = torch.matmul(cam_coords_points, R_cam_to_world.T) + t_cam_to_world


            # # 保存带颜色的点云
            color_numpy = np.array(Image.open(self.input_color_paths[frame_index]))  # 读取为HWC
            colors_rgb = color_numpy.reshape(-1, 3)  # 转换回HWC并展平
            vertices_3d = world_coords_points.reshape(-1, 3).cpu().numpy()
            point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
            point_cloud_data.export(f"{self.output_metric_pointmap_dir}/pcd_{frame_index+1:04d}.ply")
            
            
            # 保存为pointmap npy
            pointmap_data = world_coords_points.cpu().numpy()
            np.save(f"{self.output_metric_pointmap_dir}/pointmap_{frame_index+1:04d}.npy", pointmap_data)
            


        
    def render_from_cameras(self):
        render_output_dir = os.path.join(self.output_root, "rendered_views")
        os.makedirs(render_output_dir, exist_ok=True)
        
        select_frame = 0
        npy_files = sorted(
            [f for f in os.listdir(self.output_metric_pointmap_dir) if f.endswith(".npy")],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )        

        npy_path = os.path.join(self.output_metric_pointmap_dir, npy_files[select_frame])
        

        # 读取npy_path
        pointmap = np.load(npy_path)
        points = pointmap.reshape(-1, 3)
        
        color_numpy = np.array(Image.open(self.input_color_paths[select_frame]))  # 读取为HWC
        colors_rgb = color_numpy.reshape(-1, 3)  # 转换回HWC并展平
        colors = colors_rgb[:, :3]

        height, width = cv2.imread(self.input_color_paths[0]).shape[:2]
        renders, masks, _ = render_from_cameras_videos(
            points, colors, self.metric_w2c, self.metric_intrinsic, height, width
        )

        # 使用imageio保存所有结果
        for i, (render, mask) in enumerate(zip(renders, masks)):
            # 保存渲染图
            render_path = os.path.join(render_output_dir, f"render_{i:04d}.png")
            imageio.imwrite(render_path, render)
            
            # 保存掩码图
            mask_path = os.path.join(render_output_dir, f"mask_{i:04d}.png")
            imageio.imwrite(mask_path, mask)
        
        print(f"All results saved to: {render_output_dir}")
            

            


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Depth alignment and metric processing.")
    parser.add_argument('--image_dir', type=str, required=True, help='Input RGB directory')
    parser.add_argument('--moge_depth_dir', type=str, required=True, help='MOGe depth directory')
    parser.add_argument('--vggt_depth_dir', type=str, required=True, help='VGGT depth directory')
    parser.add_argument('--metric3d_depth_dir', type=str, required=True, help='Metric3D depth directory')
    parser.add_argument('--vggt_camera_json_file', type=str, required=True, help='VGGT camera JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output root directory')
    args = parser.parse_args()

    depth_align_processor = DepthAlignMetric(
        input_rgb_dir=args.image_dir,
        moge_depth_dir=args.moge_depth_dir,
        vggt_depth_dir=args.vggt_depth_dir,
        metric3d_depth_dir=args.metric3d_depth_dir,
        vggt_camera_json_file=args.vggt_camera_json_file,
        output_root=args.output_dir
    )

    depth_align_processor.align_depth_scale()
    depth_align_processor.load_metirc_camera_parameters()
    depth_align_processor.depth_to_pointmap()  
    depth_align_processor.render_from_cameras() 
