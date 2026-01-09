# backend/pc2mesh.py
import os
import numpy as np
import open3d as o3d
import torch
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.point_cloud import PointCloud

class PC2Mesh:
    def __init__(self, device=None, sdf_name="sdf"):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print("Loading SDF model on", self.device)
        self.sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], self.device)
        self.sdf_model.eval()
        self.sdf_model.load_state_dict(load_checkpoint(sdf_name, self.device))
        print("SDF model loaded.")

    def npz_to_pointcloud_obj(self, npz_path):
        d = np.load(npz_path)
        if 'coords' in d:
            return PointCloud.load(npz_path)
        if 'points' in d:
            pts = d['points']
        elif 'arr_0' in d:
            pts = d['arr_0']
        else:
            raise KeyError("No points-like key")
        cols = None
        for k in ('colors','rgb','rgba','cols'):
            if k in d: cols = d[k]; break
        if cols is not None:
            if cols.max() <= 1.0:
                cols = (cols * 255).astype(np.uint8)
            else:
                cols = cols.astype(np.uint8)
            try:
                pc = PointCloud(pts, colors=cols)
                return pc
            except Exception:
                return PointCloud(coords=pts, channels={'colors':cols})
        else:
            return PointCloud(pts)

    def pointcloud_to_mesh(self, npz_path, out_mesh_path="final_mesh.ply", grid_size=32, batch_size=4096):
        pc = self.npz_to_pointcloud_obj(npz_path)
        print("Running marching_cubes_mesh (this may take a while)")
        mesh = marching_cubes_mesh(
            pc=pc,
            model=self.sdf_model,
            batch_size=batch_size,
            grid_size=grid_size,
            progress=True,
        )
        os.makedirs(os.path.dirname(out_mesh_path) or ".", exist_ok=True)
        with open(out_mesh_path, "wb") as f:
            mesh.write_ply(f)
        print("Saved mesh:", out_mesh_path)
        return out_mesh_path
