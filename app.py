import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

MODEL_PATH = 'model/model_v2.h5'
MODEL_CONV_LAYER = "Convolution-4"
IMAGE_SIZE = (224, 224)
model = load_model(MODEL_PATH)
class_mapping = np.load('data/preproc_data/class_mapping.npy', allow_pickle=True).item()
reverse_class_mapping = {i: class_name for class_name, i in class_mapping.items()}

def preprocess_image(image, IMAGE_SIZE=(224, 224)):
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
    processed_image = image_array / 255.0
    return processed_image

def create_gradcam(img_array, model, layer, pred_index):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer).output, model.output])

    with tf.GradientTape() as tape:
        layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_image):
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.get_cmap("jet")
    cmap_colors = cmap(np.arange(256))[:, :3]
    cmap_heatmap = cmap_colors[heatmap]

    cmap_heatmap = array_to_img(cmap_heatmap)
    cmap_heatmap = cmap_heatmap.resize((original_image.shape[1], original_image.shape[0]))
    cmap_heatmap = img_to_array(cmap_heatmap)

    superimposed_img = cmap_heatmap * 0.3 + original_image
    superimposed_img = array_to_img(superimposed_img)
    return superimposed_img

def get_prediction_details(processed_image, model, reverse_class_mapping):
    test_image = np.expand_dims(processed_image, axis=0)
    result = model.predict(test_image)
    predicted_class_index = np.argmax(result[0])
    predicted_class = reverse_class_mapping[predicted_class_index]
    return test_image, predicted_class, predicted_class_index
    
def app():
    st.title('Traffic Sign Classification')
    st.markdown("Select an image from the list or upload your own to let the AI classify it for you!")

    image_list = {
        "Image 1": "path/to/image1.jpg",
        "Image 2": "path/to/image2.jpg",

    }

    selected_image_name = st.selectbox("Choose an image from the list:", options=list(image_list.keys()))

    if st.button("Load Selected Image"):
        image_path = image_list[selected_image_name]
        image = Image.open(image_path)
    elif (uploaded_file := st.file_uploader("Or upload an image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])):
        image = Image.open(uploaded_file)

    if 'image' in locals():
        with st.spinner('Processing image...'):
            cols = st.columns(2)
            cols[0].image(image, caption='Selected Image.', use_column_width=True)
            processed_image = preprocess_image(image)
            test_image, predicted_class, predicted_class_index = get_prediction_details(processed_image, model, reverse_class_mapping)

            heatmap = create_gradcam(test_image, model, MODEL_CONV_LAYER, predicted_class_index)
            heatmap = cv2.resize(heatmap, IMAGE_SIZE)
            test_image_plot = (test_image[0] * 255).astype(np.uint8)
            superimposed_img = overlay_heatmap(heatmap, test_image_plot)
            cols[1].image(superimposed_img, caption='Grad-CAM Heatmap Overlay.', use_column_width=True)

            st.success(f"Predicted class: **{predicted_class}**")

if __name__ == '__main__':
    app()


