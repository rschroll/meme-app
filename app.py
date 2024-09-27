import os

import diffusers
import streamlit as st
import torch
from PIL import ImageDraw, ImageFont

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

LORA_WEIGHTS = os.environ["LORA_WEIGHTS"]

@st.cache_resource
def load_model():
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    )
    pipeline.load_lora_weights(
        LORA_WEIGHTS, weight_name="pytorch_lora_weights.safetensors"
    )
    pipeline.to(device)
    return pipeline

def generate_images(prompt, pipeline, n):
    return pipeline([prompt] * n).images

def add_text_to_image(image, text, text_color="white", outline_color="black",
                      font_size=50, border_width=2, font_path="arial.ttf"):
    # Initialization
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Calculate the size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position at which to draw the text to center it
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=border_width, stroke_fill=outline_color)

def generate_memes(prompt, text, pipeline, n):
    images = generate_images(prompt, pipeline, n)
    for im in images:
        add_text_to_image(im, text)
    return images

def main():
    st.title("Diffusion Model Image Generator")

    num_images = st.sidebar.number_input(
        "Number of Images", min_value=1, max_value=10
    )
    prompt = st.sidebar.text_area("Text-to-Image Prompt")
    text = st.sidebar.text_area("Text to Display")

    generate = st.sidebar.button("Generate Images")
    if generate:
        if not prompt:
            st.error("Please enter a prompt.")
        elif not text:
            st.error("Please enter the text.")
        else:
            with st.spinner("Generating images..."):
                pipeline = load_model()
                images = generate_memes(prompt, text, pipeline, num_images)
                st.subheader("Generated images:")
                for im in images:
                    st.image(im)

if __name__ == '__main__':
    main()
