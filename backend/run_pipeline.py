# backend/run_pipeline.py
import os
from generate_image import generate_image
from image2pc import PointEWrapper
from pc2mesh import PC2Mesh

class Pipeline:
    def __init__(self, work_dir="."):
        self.work_dir = work_dir
        print("Initializing pipeline — loading heavy models (this may take a minute)...")
        # load point-e sampler (loads base/upsampler)
        self.point_e = PointEWrapper()
        # load sdf model
        self.pc2mesh = PC2Mesh()
        print("Pipeline ready.")

    def run(self, prompt: str):
        # 1. generate seed image
        img_path = generate_image(prompt, out_dir=self.work_dir)

        # 2. image -> pointcloud (.npz)
        npz_path, ply_path = self.point_e.image_to_pointcloud(img_path, out_dir=self.work_dir)

        # 3. pointcloud -> mesh
        final_mesh = os.path.join(self.work_dir, "final_mesh.ply")
        self.pc2mesh.pointcloud_to_mesh(npz_path, out_mesh_path=final_mesh)
        return final_mesh
