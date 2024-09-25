import diffusers
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

LORA_WEIGHTS = "rschroll/maya_model_v1_lora"

def load_model():
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    )
    pipeline.load_lora_weights(
        LORA_WEIGHTS, weight_name="pytorch_lora_weights.safetensors"
    )
    pipeline.to(device)
    return pipeline

def generate_images(prompt, pipeline, n):
    return pipeline([prompt] * n).images
