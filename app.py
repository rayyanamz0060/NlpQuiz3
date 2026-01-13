import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="AI Image Captioner", layout="wide")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    """
    Loads the BLIP model and processor from Hugging Face.
    This is cached to ensure it only runs once at startup.
    """
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    return processor, model

def generate_caption(image):
    """
    Processes the image and generates a text description.
    """
    processor, model = load_model()
    
    # Preprocess image and generate output
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    
    # Decode the result
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.capitalize()

# --- UI Components ---
st.title("📸 AI Image Captioning App")
st.markdown("Upload an image, snap a photo, or provide a URL to generate an AI description.")

# Sidebar for input selection
st.sidebar.header("Input Settings")
input_option = st.sidebar.radio(
    "Choose Image Source:",
    ("Upload File", "Camera Input", "Image URL")
)

raw_image = None

# --- Logic for Different Input Options ---
if input_option == "Upload File":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")

elif input_option == "Camera Input":
    camera_file = st.camera_input("Take a snapshot")
    if camera_file is not None:
        raw_image = Image.open(camera_file).convert("RGB")

elif input_option == "Image URL":
    url = st.text_input("Paste image URL here:")
    if url:
        try:
            response = requests.get(url, timeout=10)
            raw_image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")

# --- Display Result ---
if raw_image:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(raw_image, caption="Input Image", use_container_width=True)
    
    with col2:
        st.subheader("Generated Caption")
        with st.spinner("Analyzing image..."):
            try:
                caption = generate_caption(raw_image)
                st.success(caption)
            except Exception as e:
                st.error(f"Failed to generate caption: {e}")
else:
    st.info("Please provide an image using one of the options in the sidebar.")

# Footer
st.markdown("---")
st.caption("Powered by Hugging Face Transformers & Salesforce BLIP")
