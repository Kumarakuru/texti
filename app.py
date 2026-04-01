import streamlit as st
import requests
import io
from PIL import Image
import time

# --- Page Config ---
st.set_page_config(page_title="✨ Magic Image Creator", layout="centered")

st.markdown("""
<style>
    .main {background-color: #0f0f23;}
    h1 {color: #ff9ff3;}
    .stButton>button {
        background: linear-gradient(45deg, #ff9ff3, #f368e0);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-weight: bold;
        width: 100%;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Magic Image Creator")
st.caption("Direct Endpoint Connection • Public Mode")

# --- CONFIGURATION ---
# Use the exact URL from your HF dashboard
API_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"

def query_hf_endpoint(prompt_text, steps):
    # HF Endpoints typically expect "inputs" for the prompt
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "num_inference_steps": steps,
            "guidance_scale": 7.5
        }
    }
    
    # Direct POST request with no Authorization header
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        return response.content, None
    elif response.status_code == 503:
        return None, "Endpoint is warming up... try again in 30s."
    elif response.status_code == 404:
        return None, "404 Error: Ensure the Endpoint URL is exactly correct and 'Public'."
    else:
        return None, f"Error {response.status_code}: {response.text}"

# --- UI ---
prompt = st.text_area(
    "Describe your image ✨",
    placeholder="A cute red panda astronaut...",
    height=120
)

steps = st.slider("Quality (Steps)", 10, 50, 30)

if st.button("✨ Generate Image"):
    if not prompt.strip():
        st.error("Please enter a prompt!")
    else:
        with st.spinner("🎨 Connecting to Hugging Face..."):
            image_bytes, error = query_hf_endpoint(prompt, steps)
            
            if error:
                st.error(error)
            else:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    st.success("✨ Image Generated!")
                    st.image(image, use_container_width=True)
                    
                    # Download Setup
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download PNG",
                        data=buf.getvalue(),
                        file_name="generated_magic.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Failed to process image data: {e}")

st.divider()
st.caption("Running on Streamlit Cloud • Powered by HF Dedicated Endpoints")