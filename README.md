# ♻️ Smart Waste Classifier App

An AI-powered image classification web app that identifies various types of waste (e.g., plastic, metal, paper, etc.) using a MobileNetV2 model and provides actionable disposal, recycling, and environmental impact tips.

🌐 **Live Demo**: [Click to Open the App on Hugging Face Spaces](https://huggingface.co/spaces/ankitkumariiserb/waste-classifier)

---

## 📌 Overview

This project leverages transfer learning on **MobileNetV2** to classify waste materials from images with an accuracy of **93%**. It aims to assist users in understanding how to responsibly dispose of common waste items, along with insights like:

- Whether the item is **biodegradable**
- Its estimated **CO₂ impact**
- **Recyclability rate**
- Proper **disposal method**
- A helpful **eco-friendly tip**
- Direct links to **trusted government/eco sources**

---

## 🧠 Model Details

- **Architecture**: MobileNetV2 (Pretrained on ImageNet)
- **Training Dataset**: [Garbage Classification V2 – Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- **Input Shape**: 224x224x3
- **Classes**: Plastic, Glass, Metal, Paper, Cardboard, Trash, Battery, Clothes, Shoes, Biological
- **Final Accuracy**: ~93%

---

## 🚀 Features

- 🌍 Instant image classification of 10+ waste categories
- 🧾 Dynamic info panel with eco-data, tips, and more
- 🔗 External resource links for further reading
- 🎨 Simple, responsive Gradio interface
- 🧠 AI-driven predictions with confidence score

---

## 🧪 Tech Stack

- **Frontend/UI**: Gradio (Blocks API)  
- **Backend**: TensorFlow (Keras)  
- **Deployment**: Hugging Face Spaces  
- **Model Format**: `.h5` (Keras)

---

## 📚 Learning Outcomes

- Built an end-to-end machine learning web app from scratch
- Applied **transfer learning** using a lightweight CNN
- Learned **Gradio UI** design and deployment pipelines
- Practiced **Git**, GitHub project structure, and model inference handling

---

## 🏗️ Project Structure

Waste_Classifier_App/
│
├── app.py 
├── mobilenetv2_waste_classifier_final.h5 
├── requirements.txt 
├── runtime.txt 
├── README.md 
├── LICENSE 
├── model_training.ipynb 
└── .gitignore / .gitattributes

---

## 🏷️ License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Developed by [Ankit Kumar](https://github.com/AnkitKumarIISERB) — aspiring AI Engineer and environmental tech enthusiast.  
Feel free to ⭐ star the repo or open an issue for suggestions!

---

## 🔍 About

**Smart Waste Classifier** — A machine learning web app that classifies waste types (plastic, glass, metal, paper, etc.) from images using a trained MobileNetV2 model.  
Get eco-tips, disposal methods, CO₂ impact insights, and more — built with TensorFlow and Gradio.

