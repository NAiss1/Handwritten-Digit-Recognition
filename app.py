
# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image, ImageOps, ImageChops
from model import DigitClassifier

# PREPROCESSING FUNCTION
def process_image(img):
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox is None:
        return Image.new("L", (28, 28), 0)
    img = img.crop(bbox)
    img = img.resize((20, 20), resample=Image.BICUBIC)
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img, ((28 - 20) // 2, (28 - 20) // 2))
    return new_img




# LOAD MODEL
model = DigitClassifier()
model.load_state_dict(torch.load("saved_model/mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

st.title("\U0001F58C Real-time Handwritten Digit Recognition")
st.markdown("Draw a digit (0–9) below:")

# DRAW CANVAS
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# PREDICT
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = Image.fromarray((255 - img[:, :, 0]).astype("uint8")).convert("L")
    img = process_image(img)

    # Check if the image is mostly black (empty drawing)
    if np.sum(np.array(img)) < 1000:
        st.warning("Please draw a digit before predicting.")
    else:
        img_tensor = torch.tensor(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = (img_tensor - 0.1307) / 0.3081

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()

        st.subheader(f"Prediction: {pred} ({confidence * 100:.2f}%)")
