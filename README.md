# 😷 Face Mask Detection using VGG19

## 📌 Project Overview
This project implements a **Face Mask Detection system** using a **VGG19 Convolutional Neural Network (CNN)** trained from scratch. The model was trained on the [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset), and a **Streamlit web app** was built to provide real-time predictions on uploaded images.  

The system classifies whether a person in an image is wearing a **mask 😷** or **not wearing a mask ❌**.

---

## ⚙️ Features
- **VGG19 from scratch**: Custom implementation of the VGG19 architecture in TensorFlow/Keras.  
- **Kaggle Dataset Integration**: Downloads and preprocesses the dataset directly in Google Colab.  
- **Model Training**: Includes training pipeline, accuracy/loss visualization, and saving of `.keras` model.  
- **Streamlit Web App**: Simple UI where users can upload images and get predictions with confidence scores.  
- **Cross-platform**: Train in Google Colab and deploy locally with Streamlit.  

---

## 🧠 Model Architecture (VGG19)
- 19 layers deep CNN architecture.  
- 5 convolutional blocks with increasing filter depth.  
- Fully connected layers with dropout for regularization.  
- Binary output (`Mask` vs `No Mask`).  

---

## 📂 Project Structure
```
├── vgg19model.py        # Colab notebook script: dataset download, training, saving model
├── main.py              # Streamlit app: loads trained model, predicts uploaded images
├── vgg19_from_scratch.keras  # Trained model (generated after training)
└── data/                # Dataset (downloaded from Kaggle)
```

---

## 🚀 How to Run

### 1. Train the Model (Google Colab)
- Upload your `kaggle.json` API key.
- Run `vgg19model.py` in Colab to:
  - Download dataset  
  - Train VGG19  
  - Save/export model as `.keras`  

### 2. Run Streamlit App (VS Code / Local Machine)
```bash
pip install streamlit tensorflow pillow numpy
streamlit run main.py
```

Then open the local Streamlit link (e.g., `http://localhost:8501`) in your browser.

---

## 📊 Results
- Training and validation accuracy/loss plots are included.  
- Model achieves reliable classification of mask vs. no mask images.  

---

## 🔮 Future Improvements
- Use **transfer learning** with pre-trained ImageNet weights for better accuracy and faster convergence.  
- Extend dataset for **multi-class detection** (e.g., Mask, No Mask, Incorrect Mask).  
- Deploy as a **web service** (Flask/FastAPI) or **mobile app**.  

---

## 🙌 Acknowledgments
- Dataset: [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
- Model: VGG19 CNN architecture  
