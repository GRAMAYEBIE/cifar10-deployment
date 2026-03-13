import os
import random
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd
import sqlite3
import datetime

# --- 1. REPRODUCIBILITY CONFIG (Fixes shifting results) ---
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --- 2. DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect('predictions_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY, date TEXT, image_path TEXT, prediction TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

st.set_page_config(page_title="Deep Learning Strategic Dashboard", layout="wide")

# --- 3. ROBUST MODEL LOADER (Fixes the 'dense_10' 2-input error) ---
@st.cache_resource
def load_engine(arch_choice):
    # CRITICAL: Clear background memory to prevent layer name collisions
    tf.keras.backend.clear_session()
    
    # Define paths based on your renamed files
    if "Scratch" in arch_choice:
        path = 'my_scratch_model.h5'
    else:
        path = 'mobilenetv2_finetuned_model.h5'
    
    if not os.path.exists(path):
        return None

    try:
        # Load without compiling to strictly avoid the multiple input tensor error
        model = tf.keras.models.load_model(path, compile=False)
        
        # Warm-up: Lock the graph layers in memory with a dummy prediction
        dummy_input = np.zeros((1, 32, 32, 3))
        model.predict(dummy_input)
        return model
    except Exception as e:
        st.error(f"Engine Load Error: {e}")
        return None

# --- 4. USER INTERFACE ---
st.title("🚀 Deep Learning Strategic Dashboard")

with st.sidebar:
    st.header("🕹️ Controls")
    selected_arch = st.radio("Choose Model Architecture:", ["Custom Scratch CNN", "MobileNetV2 Transfer"])
    
    if st.button("♻️ Reset Engine & Cache"):
        st.cache_resource.clear()
        st.rerun()

# Execute Engine Loading
active_model = load_engine(selected_arch)

if active_model:
    st.sidebar.success(f"{selected_arch} Engine Online")
else:
    st.sidebar.error("Model file not found!")
    st.warning(f"Make sure {selected_arch} .h5 file is in the root directory.")

# --- 5. INFERENCE WORKFLOW ---
uploaded_file = st.file_uploader("Upload an asset for analysis...", type=["jpg", "png", "jpeg"])

if uploaded_file and active_model:
    # Asset storage
    if not os.path.exists("stored_images"): os.makedirs("stored_images")
    img_path = os.path.join("stored_images", uploaded_file.name)
    with open(img_path, "wb") as f: f.write(uploaded_file.getbuffer())

    # Pre-processing (CIFAR-10 requirements)
    raw_img = Image.open(uploaded_file)
    img_resized = raw_img.resize((32, 32))
    img_tensor = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🖼️ Input Asset")
        st.image(raw_img, use_container_width=True)

    with col2:
        st.subheader("📊 Inference & Intelligence")
        if st.button("EXECUTE ANALYSIS", type="primary"):
            # Run prediction
            preds = active_model.predict(img_tensor)[0]
            # Manual softmax for probability stability
            probs = tf.nn.softmax(preds).numpy()
            
            classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                       'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            top_idx = np.argmax(probs)
            label, conf = classes[top_idx], float(probs[top_idx])

            # Result Metrics
            st.metric(label="Predicted Outcome", value=label, delta=f"{conf*100:.2f}% Confidence")
            
            # Probability Distribution Chart
            res_df = pd.DataFrame({'Class': classes, 'Probability': probs}).sort_values('Probability', ascending=False)
            st.bar_chart(res_df, x='Class', y='Probability', color="#FF4B4B")

            # Database logging
            conn = sqlite3.connect('predictions_history.db')
            c = conn.cursor()
            c.execute("INSERT INTO history (date, image_path, prediction, confidence) VALUES (?, ?, ?, ?)",
                      (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_path, label, conf))
            conn.commit()
            conn.close()

# --- 6. LOGS ---
st.divider()
if st.checkbox("Show History Logs"):
    if os.path.exists('predictions_history.db'):
        conn = sqlite3.connect('predictions_history.db')
        st.dataframe(pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn), use_container_width=True)
        conn.close()