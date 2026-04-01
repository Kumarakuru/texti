import streamlit as st
from openai import OpenAI
import time
from PIL import Image
import io

# Page Config
st.set_page_config(page_title="✨ Magic Image Creator", layout="centered")

# Custom CSS for the "Magic" look
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
        border: none;
    }
    .stButton>button:hover {
        border: 1px solid #ff9ff3;
        color: #ff9ff3;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Magic Image Creator")
st.caption("Text to Image • Powered by Hugging Face Dedicated Endpoints")

# --- CONFIGURATION ---
# Removed /v1 to let the OpenAI SDK handle path construction correctly
IMAGE_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"
# Use your actual HF Token here if 'hf_dummy' fails
HF_TOKEN = "hf_dummy" 

client = OpenAI(base_url=IMAGE_URL, api_key=HF_TOKEN)

# --- HELPERS ---
def wait_for_hf_endpoint(fn, label="Image API", max_wait=180, interval=25):
    start = time.time()
    attempt = 0
    while True:
        try:
            result = fn()
            if attempt > 0:
                st.success(f"✅ {label} is ready!")
            return result, None
        except Exception as e:
            err_msg = str(e)
            elapsed = int(time.time() - start)
            
            # 503 means the endpoint is sleeping/warming up
            if "503" not in err_msg and "404" not in err_msg:
                return None, f"Error: {err_msg}"
            
            if elapsed >= max_wait:
                return None, f"Endpoint did not wake up after {max_wait//60} minutes."
            
            attempt += 1
            st.warning(f"⏳ {label} warming up... (attempt {attempt})")
            time.sleep(interval)

# --- UI ---
prompt = st.text_area(
    "Describe your image ✨",
    placeholder="A cute red panda astronaut floating in space, colorful nebula background, highly detailed",
    height=120
)

col1, col2 = st.columns(2)
with col1:
    n_images = st.slider("Number of images", 1, 4, 1)
with col2:
    steps = st.slider("Quality (Steps)", 20, 50, 30)

if st.button("✨ Generate Images", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please describe the image!")
    else:
        with st.spinner("🎨 Generating your images..."):
            
            def generate_call():
                # 'model' is required for the SDK to route the request correctly
                # 'extra_body' handles non-standard parameters like steps
                return client.images.generate(
                    model="timbrooks/instruct-pix2pix", 
                    prompt=prompt,
                    n=n_images,
                    response_format="b64_json",
                    extra_body={
                        "num_inference_steps": steps
                    }
                )

            response, error = wait_for_hf_endpoint(generate_call, label="Image Generation")

            if error:
                st.error(f"Failed to connect: {error}")
                st.info("Check if your Endpoint URL is correct and the status is 'Running' in Hugging Face.")
            else:
                st.success("✨ Here are your images!")

                for i, img_data in enumerate(response.data):
                    try:
                        # Decode the base64 response
                        import base64
                        image_bytes = io.BytesIO(base64.b64decode(img_data.b64_json))
                        image = Image.open(image_bytes)
                        
                        st.image(image, caption=f"Image {i+1}", use_container_width=True)
                        
                        # Prepare download button
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
                        st.error(f"Failed to display image {i+1}: {e}")

st.divider()
st.caption("Simple & cute Text-to-Image app • Hosted on Streamlit Cloud")