# 🏷️ AutoLabel AI – Local Image Annotation Tool

AutoLabel AI is a local web-based image annotation tool that helps speed up dataset creation for machine learning. It supports **manual bounding box labeling** and optional **AI-assisted auto-labeling**, similar to tools like Roboflow.

---

## 🚀 Features

- 📤 Upload images for labeling
- 🖱️ Draw bounding boxes manually
- 🏷️ Assign class labels to objects
- 🤖 Auto-label support (if AI model is integrated)
- ✏️ Edit / delete annotations easily
- 💾 Save labels locally
- 📦 Export dataset in YOLO format
- 🌐 Fully runs on local system (no internet required)

---

## 🧠 Tech Stack

- Python (Flask)
- HTML / CSS / JavaScript
- Optional AI model integration (YOLO or similar)
- Local file-based storage

---

## 📁 Project Structure


auto-labeler/
│
├── app.py # Flask backend server
├── templates/
│ └── index.html # Main frontend UI
├── static/ # CSS, JS, assets (if used)
├── venv/ # Virtual environment (NOT included in submission)
└── README.md


---

## ⚙️ How to Run the Project

### 1. Open project folder
```bash
cd auto-labeler
2. Create virtual environment (if not already created)
python3 -m venv venv
source venv/bin/activate
3. Install required dependencies
pip install flask

(If you used additional libraries like OpenCV or YOLO, install them too)

4. Run the backend server
python app.py