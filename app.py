import streamlit as st
import requests
import time
from PIL import Image
import io
import base64

st.set_page_config(page_title="✨ Magic Image Creator", layout="centered")

st.markdown("""
<style>
    .main {background-color: #0f0f23;}
    h1 {color: #ff9ff3;}
    .stButton>button {
        background: linear-gradient(45deg, #ff9ff3, #f368e0);
        color: white;
        border-radius: 12px;
        height: 3.2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Magic Image Creator")
st.caption("Text to Image • Powered by Hugging Face")

# Your Exact Endpoint
ENDPOINT_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud/v1/images/generations"

prompt = st.text_area(
    "Describe your image ✨",
    placeholder="A cute red panda wearing sunglasses and riding a skateboard in a neon city at night",
    height=120
)

col1, col2 = st.columns(2)
with col1:
    n_images = st.slider("Number of images", 1, 4, 2)
with col2:
    steps = st.slider("Quality (Steps)", 20, 50, 30)

if st.button("✨ Generate Images", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please describe the image!")
    else:
        with st.spinner("🎨 Generating your images... Please wait"):
            try:
                payload = {
                    "prompt": prompt,
                    "num_images": n_images,
                    "num_inference_steps": steps,
                    "guidance_scale": 7.5,
                    "response_format": "b64_json"
                }

                response = requests.post(
                    ENDPOINT_URL,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✨ Here are your images!")

                    # Handle different possible response formats
                    images_list = result.get("images") or result.get("data") or [result]

                    for i, item in enumerate(images_list[:n_images]):
                        try:
                            b64_str = item.get("b64_json") or item.get("image") or item
                            image_bytes = base64.b64decode(b64_str)
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            st.image(image, caption=f"Image {i+1}", use_column_width=True)
                            
                            buf = io.BytesIO()
                            image.save(buf, format="PNG")
                            st.download_button(
                                label=f"⬇️ Download Image {i+1}",
                                data=buf.getvalue(),
                                file_name=f"magic_image_{i+1}.png",
                                mime="image/png",
                                key=f"dl_{i}"
                            )
                        except Exception as e:
                            st.error(f"Failed to display image {i+1}")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Request failed: {str(e)}")

st.caption("Simple Text-to-Image app • Hosted on Streamlit Cloud")