import streamlit as st
import onnxruntime
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

# Load ONNX model
ort_session = onnxruntime.InferenceSession("generator.onnx")

# Create a session state to store the generated image
class SessionState:
    def __init__(self):
        self.img = None

state = SessionState()

def generate(red=10, green=10, blue=10, saturation=2.0, contrast=2.0):
    # Generate image using the ONNX model
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    ort_inputs = {input_name: np.random.randn(1, 128, 1, 1).astype(np.float32)}
    output = ort_session.run([output_name], ort_inputs)
    output_np = output[0].squeeze().transpose((1, 2, 0))

    # Convert pixel values to the range [0, 1]
    output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min())

    # Upscale the image
    upscaled_img = cv2.resize(output_np, (2160, 4096), interpolation=cv2.INTER_LINEAR)

    def posterize_channel(channel, num_colors):
        quantized_values = np.linspace(0, 1, num_colors)
        return np.digitize(channel, quantized_values) * (1 / num_colors)

    # Split the image into RGB channels
    r, g, b = upscaled_img[:, :, 0], upscaled_img[:, :, 1], upscaled_img[:, :, 2]

    # Posterize each channel with specified number of colors
    num_colors_red = red
    num_colors_green = green
    num_colors_blue = blue

    posterized_r = posterize_channel(r, num_colors_red)
    posterized_g = posterize_channel(g, num_colors_green)
    posterized_b = posterize_channel(b, num_colors_blue)

    # Combine the posterized channels back into a single image
    merged_array = np.dstack((posterized_r, posterized_g, posterized_b))
    posterized_image = Image.fromarray((merged_array * 255).astype(np.uint8))

    # Enhance the color saturation
    enhancer = ImageEnhance.Color(posterized_image)
    boosted_image = enhancer.enhance(saturation)  # Increase the saturation

    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(boosted_image)
    high_contrast_image = enhancer.enhance(contrast)  # Increase the contrast

    return high_contrast_image

def main():
    st.title("GAN WALLPAPER GENERATOR")
    col1, col2 = st.columns([3, 2], gap="small")
    
    with col2:
        red = st.slider("Red Importance", min_value=1, max_value=50, value=40)
        green = st.slider("Green Importance", min_value=1, max_value=50, value=40)
        blue = st.slider("Blue Importance", min_value=1, max_value=50, value=40)
        saturation = st.slider("Saturation", min_value=1.0, max_value=10.0, value=2.0)
        contrast = st.slider("Contrast", min_value=1.0, max_value=10.0, value=2.0)
    with col1:
        state.img = generate(abs(red-49), abs(green-49), abs(blue-49), saturation, contrast)
        st.image(state.img, caption=None, use_column_width=False, width=300)
        # # Add buttons below the sliders
        # col1.button("Download Image")
        # col1.button("Generate New Image")
    # cols1, cols2 = st.columns([0.7, 0.3], gap=0)
    button_generate = st.button("Generate")
    def download_image(img):
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        st.download_button(
            label="Download Image",
            data=img_bytes,
            file_name='image.jpg',
            mime='image/jpeg'
        )

    download_image(state.img)
    if button_generate:
        # Generate the image
        state.img = generate(abs(red-49), abs(green-49), abs(blue-49), saturation, contrast)

if __name__ == "__main__":
    main()
