# backend/generate_image.py
import os
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
SAVE_NAME = "seed_view.png"

def generate_image(prompt: str, out_dir: str = ".") -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device for image generation:", device)

    # load pipeline (lightweight config as in your notebook)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )

    # low VRAM settings:
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()

    # generate 256x256 clay-like image (you can parameterize this)
    out = pipe(
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=256,
        width=256,
    )
    img = out.images[0]
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, SAVE_NAME)
    img.save(save_path)
    print("Saved seed image:", save_path)
    return save_path
