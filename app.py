import streamlit as st
import requests
import io
import time
import base64

# --- Page Setup ---
st.set_page_config(page_title="✨ Magic Media Creator", layout="centered")

# Configuration for separate endpoints
IMAGE_API_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud"
# Replace with your Video Endpoint URL if different
VIDEO_API_URL = "https://ecqhp03p8738815i.us-east-1.aws.endpoints.huggingface.cloud" 

def handle_response(response):
    """Decodes image or video data from binary or base64."""
    content_type = response.headers.get("Content-Type", "")
    
    if response.status_code == 200:
        if "image" in content_type or "video" in content_type:
            return response.content, None
        
        # Handle Base64 strings (Common for HF JSON responses)
        try:
            data = response.text.strip('"').strip("'")
            if data.startswith("iVBORw0") or data.startswith("AAAAIGZ0eX"): # PNG or MP4 headers
                return base64.b64decode(data), None
            return None, response.json()
        except:
            return None, response.text
    return None, f"Status {response.status_code}: {response.text}"

# --- UI ---
st.title("✨ Magic Media Creator")
tab1, tab2 = st.tabs(["🖼️ Image Generation", "🎥 Video Generation"])

with tab1:
    prompt_img = st.text_area("Image Prompt", value="adiyogi shiva, masterpiece, cinematic lighting", key="img_p")
    if st.button("Generate Image", use_container_width=True):
        with st.spinner("🎨 Generating..."):
            res, err = handle_response(requests.post(IMAGE_API_URL, json={"inputs": prompt_img}))
            if err: st.error(err)
            else: st.image(res, use_container_width=True)

with tab2:
    st.info("Video generation takes 1-2 minutes per clip.")
    prompt_vid = st.text_area("Video Prompt", value="adiyogi statue with clouds moving in the background, hyper-realistic", key="vid_p")
    
    if st.button("Generate Video", use_container_width=True):
        with st.spinner("🎬 Rendering Video (This takes a while)..."):
            # Note: Many HF Video endpoints require an 'image' input to animate, 
            # but some text-to-video models take 'inputs' directly.
            payload = {
                "inputs": prompt_vid,
                "parameters": {"fps": 6, "motion_bucket_id": 127}
            }
            
            try:
                # Use a much longer timeout for video
                response = requests.post(VIDEO_API_URL, json=payload, timeout=300)
                res, err = handle_response(response)
                
                if err:
                    st.error(f"Video Error: {err}")
                else:
                    st.success("🎥 Video Ready!")
                    st.video(res)
                    st.download_button("⬇️ Download Video", res, "generated_video.mp4", "video/mp4")
            except Exception as e:
                st.error(f"Request Timed Out: {e}. Video generation often exceeds standard web timeouts.")

st.divider()
st.caption("Note: Video generation requires a high-VRAM GPU (like A100 or L4) on your Hugging Face Endpoint.")