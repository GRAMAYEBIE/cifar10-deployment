# 🚀 CIFAR-10 Strategic Deep Learning Dashboard

This project features an interactive dashboard designed to compare two Deep Learning approaches for image classification (CIFAR-10): a **Custom CNN** (built from scratch) and **Transfer Learning via MobileNetV2**.

## 📊 Project Overview
The goal is to provide a real-time diagnostic interface where users can upload an image, select a model architecture, and observe the prediction probability distribution.

### Key Features:
* **Dual-Architecture Support**: Seamlessly switch between a lightweight Custom CNN and a high-performance pre-trained MobileNetV2.
* **Robust Inference Engine**: Resolved Keras tensor conflicts (specifically the `dense_10` input error) using non-compiled model loading techniques.
* **Data Persistence**: Built-in prediction logging using a **SQLite** database to track historical outcomes.
* **Dynamic UI**: Modern Streamlit dashboard with real-time probability bar charts and performance metrics.

## 🛠️ Technical Stack
* **Framework**: TensorFlow / Keras
* **Models**: 
    1.  *Custom CNN*: 3 Convolutional layers, Max Pooling, and Dropout (Trained for 50 epochs).
    2.  *MobileNetV2*: Fine-tuned architecture with base layer freezing for optimal feature extraction.
* **Deployment**: Streamlit Cloud (optimized with `tensorflow-cpu`).
* **Reproducibility**: Global seeding (NumPy, TF, Random) to ensure consistent inference scores across sessions.

## 🚀 Installation & Deployment

### Local Environment
1. Clone the repository:
   ```bash
   git clone [https://github.com/GRAMAYEBIE/cifar10-deployment.git](https://github.com/GRAMAYEBIE/cifar10-deployment.git)

2. Install dependencies:

Bash
pip install -r requirements.txt

3. Run the application:

Bash
streamlit run app.py
Cloud Deployment
This app is ready for Streamlit Cloud. It utilizes the tensorflow-cpu build to remain within the resource limits of the cloud environment while maintaining fast inference times.

📂 Repository Structure
app.py: The core Streamlit application logic.

my_scratch_model.h5: Trained weights for the Custom CNN model.

mobilenetv2_finetuned_model.h5: Cleaned weights for the MobileNetV2 model.

requirements.txt: List of Python dependencies.

.gitignore: Configuration to exclude virtual environments (venv/) and local caches.

🧠 Technical Challenges Solved
Memory Management: Implemented tf.keras.backend.clear_session() to prevent layer name collisions when switching models in the UI.

Tensor Input Bug: Fixed the ValueError: Layer dense_10 expects 1 input, but received 2 by loading models with compile=False, bypassing corrupted optimizer states in the .h5 files.

CI/CD Optimization: Configured .gitignore to prevent pushing heavy venv folders (600MB+), reducing deployment time and adhering to GitHub's file size limits.

Developed by GRAMAYEBIE Master’s Candidate in Data Science & Analytics
