# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from run_pipeline import Pipeline
import os, threading

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://10.95.1.117:5173"],  # adjust frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str

# Initialize pipeline at startup (loads models once)
pipeline = Pipeline(work_dir=os.path.abspath("."))

@app.post("/generate")
def generate(prompt: Prompt):
    # blocking call - runs pipeline synchronously
    try:
        mesh_path = pipeline.run(prompt.prompt)
        # return a direct mesh endpoint
        return {"mesh_url": f"http://localhost:5000/mesh"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mesh")
def get_mesh():
    path = os.path.abspath("final_mesh.ply")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mesh not found")
    return FileResponse(path, media_type="application/octet-stream", filename="final_mesh.ply")
