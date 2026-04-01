import streamlit as st
from openai import OpenAI
import time
from PIL import Image
import io
import base64
import requests

st.set_page_config(page_title="✨ Magic Image Creator", layout="centered")

# --- CONFIGURATION ---
# Base URL for Hugging Face Dedicated Endpoints
IMAGE_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = "hf_dummy" # Replace with your actual token if needed

client = OpenAI(base_url=IMAGE_URL, api_key=HF_TOKEN)

st.title("✨ Magic Image Creator")

# UI Elements
prompt = st.text_area("Describe your image ✨", value="A cute red panda")
steps = st.slider("Quality (Steps)", 20, 50, 30)

if st.button("✨ Generate & Debug", type="primary"):
    # --- DEBUGGING INFO ---
    # This is the path the OpenAI SDK constructs internally
    target_url = f"{IMAGE_URL}/images/generations"
    st.info(f"Connecting to: `{target_url}`")
    
    with st.spinner("🎨 Generating..."):
        try:
            # We use extra_body to pass custom HF parameters
            response = client.images.generate(
                model="timbrooks/instruct-pix2pix", 
                prompt=prompt,
                n=1,
                response_format="b64_json",
                extra_body={"num_inference_steps": steps}
            )
            
            # If successful, display image
            img_b64 = response.data[0].b64_json
            image = Image.open(io.BytesIO(base64.b64decode(img_b64)))
            st.image(image, caption="Generated Image")
            
        except Exception as e:
            st.error(f"Error Type: {type(e).__name__}")
            st.error(f"Message: {str(e)}")
            
            # Additional check: Is the endpoint even reachable?
            st.markdown("---")
            st.subheader("Network Check")
            try:
                health_check = requests.get(IMAGE_URL)
                st.write(f"Endpoint Root Status: {health_check.status_code}")
                if health_check.status_code == 404:
                    st.warning("The root URL exists but the path /images/generations might not be enabled on this HF endpoint.")
            except Exception as net_err:
                st.error(f"Network Connection Failed: {net_err}")