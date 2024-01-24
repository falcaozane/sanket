import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# Load the RGB to hyperspectral conversion model
conversion_model = tf.keras.models.load_model("model.h5")

def convert_to_image(tensor):
    # Normalize the tensor values between 0 and 1
    tensor_min = np.min(tensor)
    tensor_max = np.max(tensor)
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    # Convert the normalized tensor to a uint8 image
    img_array = (normalized_tensor * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    return img

def convert_and_display(rgb_image):
    # Resize the RGB image
    img = Image.fromarray(rgb_image.astype('uint8'), 'RGB')
    img = img.resize((272, 512))
    arr = np.array(img).astype('float32') / 255.0
    new_size = (272, 512)
    resized_rgb_img = tf.image.resize(arr, new_size)
    resized_rgb_img = tf.reshape(resized_rgb_img, (1, 272, 512, 3))  # Add batch dimension

    # Convert the entire RGB image to hyperspectral using the model
    hyperspectral_image = conversion_model(resized_rgb_img)
    hyperspectral_image = tf.image.resize(hyperspectral_image, new_size)
    hyperspectral_image_np = hyperspectral_image.numpy()

    # Prepare the hyperspectral image for display
    hyperspectral_image_display = convert_to_image(hyperspectral_image_np[0])

    return hyperspectral_image_display

# Define the Gradio interface
image_input = gr.Image()
gr.Interface(
    fn=convert_and_display,
    inputs=image_input,
    outputs=gr.Image(),
    live=True,
    title="RGB to Hyperspectral Conversion",
    description="Upload an RGB image and view the converted hyperspectral image."
).launch()
