# backend/image2pc.py
import os
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from tqdm import tqdm

# point-e imports (from your notebook)
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

# Paths default (adjust if needed)
DEFAULT_DIR = "."
LOW_RES_NPZ = "cloud_1024.npz"
UPSAMPLED_PLY = "cloud_4096.ply"

class PointEWrapper:
    def __init__(self, device=None, base_name="base40M", upsample_name="upsample"):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print("PointE device:", self.device)
        # build models once
        print("Creating base model...")
        self.base_model = model_from_config(MODEL_CONFIGS[base_name], self.device)
        self.base_model.eval()
        self.base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        print("Creating upsample model...")
        self.upsampler_model = model_from_config(MODEL_CONFIGS[upsample_name], self.device)
        self.upsampler_model.eval()
        self.upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsample_name])

        print("Loading checkpoints...")
        self.base_model.load_state_dict(load_checkpoint(base_name, self.device))
        self.upsampler_model.load_state_dict(load_checkpoint(upsample_name, self.device))
        print("PointE models loaded.")

        self.sampler = PointCloudSampler(
            device=self.device,
            models=[self.base_model, self.upsampler_model],
            diffusions=[self.base_diffusion, self.upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=["R", "G", "B"],
            guidance_scale=[3.0, 3.0],
        )
        print("Sampler created.")

    def image_to_pointcloud(self, image_path: str, out_dir: str = DEFAULT_DIR):
        img = Image.open(image_path).convert("RGB").resize((256,256))
        samples = None
        for out in tqdm(self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = out
        pcs = self.sampler.output_to_point_clouds(samples)
        pc = pcs[0]

        # extract pts & cols (similar to your helper)
        def extract_pc(obj):
            for name in ["points","xyz","verts","vertices"]:
                if hasattr(obj, name):
                    pts = np.asarray(getattr(obj, name))
                    if pts.ndim == 2 and pts.shape[1] >=3:
                        break
            else:
                found = None
                for k,v in obj.__dict__.items():
                    arr = np.asarray(v)
                    if arr.ndim == 2 and arr.shape[1] >=3:
                        pts = arr; found = True; break
                if not found:
                    raise AttributeError("Could not find points in pc object.")
            cols = None
            for name in ["colors","rgb","rgba"]:
                if hasattr(obj, name):
                    cols = np.asarray(getattr(obj, name)); break
            if cols is None:
                cols = np.ones_like(pts) * 128
            else:
                if cols.max() <= 1.0:
                    cols = (cols * 255).astype(np.uint8)
                else:
                    cols = cols.astype(np.uint8)
            return pts[:,:3], cols[:,:3]

        pts, cols = extract_pc(pc)
        npz_path = os.path.join(out_dir, LOW_RES_NPZ)
        ply_path = os.path.join(out_dir, UPSAMPLED_PLY)
        np.savez_compressed(npz_path, points=pts, colors=cols)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(float) / 255.0)
        o3d.io.write_point_cloud(ply_path, pcd)
        print("Saved point cloud:", npz_path, ply_path)
        return npz_path, ply_path
