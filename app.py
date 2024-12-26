import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import RRDBNet_arch as arch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# Set up Streamlit page configuration
st.set_page_config(page_title="Photo Editor with Metrics", layout="wide")

# Define the model path
MODEL_PATH = './models/RRDB_ESRGAN_x4.pth'

# Load the model
@st.cache_resource
def load_model():
    device = torch.device('cpu')  # Use CPU for inference
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=True)
    model.eval()
    model = model.to(device)
    return model, device

model, device = load_model()

# Functions for image processing
def upscale_image(image):
    img = np.array(image) / 255.0  # Normalize
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) * 255.0
    return output.round().astype(np.uint8)

def reduce_noise(image):
    image = np.array(image)  # Convert PIL image to NumPy array
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return denoised_image

def apply_thresholding(image, threshold=127):
    image = np.array(image.convert("L"))  # Convert PIL image to grayscale
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresholded_image

def crop_image(image, left, top, right, bottom):
    image = np.array(image)
    cropped_image = image[top:bottom, left:right]
    return cropped_image

# Compute metrics
def compute_metrics(original, processed):
    """
    Computes PSNR, SSIM, and MSE between two images.
    Dynamically adjusts win_size for SSIM to avoid exceeding image dimensions.
    """
    original = np.array(original)
    processed = Image.fromarray(np.array(processed)).resize(original.shape[1::-1])  # Resize to match original dimensions
    processed = np.array(processed)

    # Determine appropriate win_size
    smaller_dim = min(original.shape[0], original.shape[1])
    win_size = smaller_dim if smaller_dim % 2 == 1 else smaller_dim - 1  # Ensure win_size is odd

    # Compute metrics
    psnr_value = psnr(original, processed, data_range=original.max() - original.min())
    ssim_value = ssim(original, processed, channel_axis=-1, win_size=win_size, data_range=original.max() - original.min())
    mse_value = mse(original, processed)

    return {"PSNR": psnr_value, "SSIM": ssim_value, "MSE": mse_value}

# Streamlit UI
st.title("Advanced Photo Editor with Metrics")
st.write("Upload an image, apply enhancements, and evaluate their quality using PSNR, SSIM, and MSE.")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Load the uploaded image
    if "original_image" not in st.session_state:
        st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.working_image = st.session_state.original_image

    # Sidebar for effects
    st.sidebar.header("Apply Effects")

    # Noise Reduction
    if st.sidebar.button("Apply Noise Reduction"):
        st.session_state.working_image = Image.fromarray(reduce_noise(st.session_state.working_image))
        st.success("Noise reduction applied.")

    # Upscaling
    if st.sidebar.button("Upscale Image"):
        st.session_state.working_image = Image.fromarray(upscale_image(st.session_state.working_image))
        st.success("Image upscaled.")

    # Thresholding
    st.sidebar.subheader("Thresholding")
    threshold_value = st.sidebar.slider("Set Threshold Value", min_value=0, max_value=255, value=127, step=1, key="threshold_slider")
    if st.sidebar.button("Apply Thresholding"):
        st.session_state.working_image = Image.fromarray(apply_thresholding(st.session_state.working_image, threshold=threshold_value))
        st.success(f"Thresholding applied with threshold = {threshold_value}.")

    # Cropping
    st.sidebar.subheader("Cropping")
    left = st.sidebar.number_input("Left", min_value=0, max_value=st.session_state.working_image.size[0], value=0, step=1)
    top = st.sidebar.number_input("Top", min_value=0, max_value=st.session_state.working_image.size[1], value=0, step=1)
    right = st.sidebar.number_input("Right", min_value=left + 1, max_value=st.session_state.working_image.size[0], value=st.session_state.working_image.size[0], step=1)
    bottom = st.sidebar.number_input("Bottom", min_value=top + 1, max_value=st.session_state.working_image.size[1], value=st.session_state.working_image.size[1], step=1)

    if st.sidebar.button("Apply Cropping"):
        st.session_state.working_image = Image.fromarray(crop_image(st.session_state.working_image, left, top, right, bottom))
        st.success("Cropping applied.")

    # Metrics Section
    st.sidebar.header("Image Quality Metrics")
    if st.sidebar.button("Compute Metrics"):
        if "original_image" in st.session_state and "working_image" in st.session_state:
            metrics = compute_metrics(st.session_state.original_image, st.session_state.working_image)
            st.sidebar.write("### Metrics")
            st.sidebar.write(f"**PSNR:** {metrics['PSNR']:.2f}")
            st.sidebar.write(f"**SSIM:** {metrics['SSIM']:.4f}")
            st.sidebar.write(f"**MSE:** {metrics['MSE']:.2f}")
        else:
            st.sidebar.warning("Original or processed image is missing!")

    # Display side-by-side images
    col1, col2 = st.columns(2)

    with col1:
        st.image(st.session_state.original_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(st.session_state.working_image, caption="Processed Image", use_container_width=True)

    # Final download button
    st.sidebar.download_button(
        label="Download Processed Image",
        data=st.session_state.working_image.tobytes(),
        file_name="processed_image.png",
        mime="image/png"
    )
