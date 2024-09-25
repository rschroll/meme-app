import diffusers
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_model():
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    )
    pipeline.to(device)
    return pipeline
