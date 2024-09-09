import streamlit as st
from diffusers import DiffusionPipeline
import torch

# Load the diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipeline.to("cuda")  # Move the pipeline to GPU if available

# Streamlit UI
st.set_page_config(page_title="Zeroscope Video Generation")
st.title("Zeroscope Video Generation")

# Input fields for user prompt and video settings
prompt = st.text_area("Enter your prompt:", height=100)
num_frames = st.number_input("Number of frames:", min_value=1, max_value=100, value=16, step=1)
width = st.number_input("Width:", min_value=64, max_value=576, value=576, step=32)
height = st.number_input("Height:", min_value=64, max_value=576, value=576, step=32)

if st.button("Generate Video"):
    if prompt:
        with st.spinner("Generating video..."):
            # Generate a video based on the prompt
            video = pipeline(prompt, num_inference_steps=num_frames, width=width, height=height).videos[0]
            
            # Display the generated video
            st.video(video)
    else:
        st.error("Please enter a prompt.")
