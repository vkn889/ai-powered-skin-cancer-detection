
import streamlit as st
from joblib import load
import numpy as np
import cv2

model = load("vgg_model.joblib")

st.title('Skin Cancer Detection')
uploaded_file = st.file_uploader("Upload Image")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Resize to 224x224 (keep color channels)
    rescaled_img = cv2.resize(image, (224,224))
    
    # Normalize pixel values (if your model was trained with normalization)
    rescaled_img = rescaled_img.astype('float32') / 255.0
    
    # Add batch dimension: shape becomes (1, 224, 224, 3)
    model_input = np.expand_dims(rescaled_img, axis=0)
    
    # Make prediction
    prediction = model.predict(model_input)
    
    # Extract the actual prediction value
    pred_value = prediction[0][0]  # Get the first prediction, first value
    confidence = float(pred_value)  # Convert to Python float
    
    # Display result
    if confidence > 0.5:
        st.write("**Result: Malignant** ⚠️")
        confidence_pct = confidence * 100
        st.write(f"Confidence: {confidence_pct:.1f}%")
    else:
        st.write("**Result: Benign** ✅")
        confidence_pct = (1 - confidence) * 100
        st.write(f"Confidence: {confidence_pct:.1f}%")
