import numpy as np
import spectral
from PIL import Image
import streamlit as st

def spectral_unmixing_simulation(rgb_image):
    # For simplicity, using example endmembers (red, green, blue)
    endmembers = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # Unmixing using the unmix function
    unmixed_data = spectral.unmix(rgb_image, endmembers)

    # Normalize the simulated hyperspectral data
    normalized_data = (unmixed_data - np.min(unmixed_data)) / (np.max(unmixed_data) - np.min(unmixed_data))

    # Convert to uint8 for display
    hyperspectral_image = (normalized_data * 255).astype(np.uint8)

    return hyperspectral_image

def main():
    st.title("RGB to Hyperspectral Image Conversion")

    uploaded_file = st.file_uploader("Upload an RGB image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded RGB Image", use_column_width=True)

        # Load and process the image
        rgb_image = np.array(Image.open(uploaded_file))
        hyperspectral_image = spectral_unmixing_simulation(rgb_image)

        # Display the simulated hyperspectral image
        st.image(hyperspectral_image, caption="Simulated Hyperspectral Image", use_column_width=True)

if __name__ == "__main__":
    main()
