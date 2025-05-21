# 🌿 FloraAI – Medicinal Plant Detection

FloraAI is a deep learning-powered medicinal plant classification web app. It helps users identify medicinal plants from images using a trained PyTorch model, and provides detailed descriptions of the plants using the OpenAI GPT API.

---

## 🧠 Features

- 🔍 Detect medicinal plants from uploaded images
- 🤖 Pretrained PyTorch model for image classification
- 🧠 OpenAI GPT integration to describe the plant uses
- 🖥️ Flask backend and HTML frontend
- 📁 Supports large datasets with Git LFS for model storage

---

## 📥 Dataset

Download the **Medicinal Plant Dataset** from [Kaggle](https://www.kaggle.com/).  
You can search for terms like `medicinal plant classification`.

Organize it in this format:
dataset/
├── Aloe_Vera/
├── Basil/
├── Neem/
└── ... (other plant folders)

yaml
Copy
Edit

---

## 🧪 Model Training

The model is trained using PyTorch (ResNet18 or any CNN). You can retrain if needed.


python train.py --data_dir dataset --epochs 25 --save_model model/medicinal_classifier2.pth
Make sure the model is saved in the model/ directory.

⚠️ Use Git LFS to track .pth files if the model is larger than 100 MB:
### 🔁 Training Command:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add model/medicinal_classifier2.pth
git commit -m "Add model with Git LFS"
```

🔑 OpenAI API Key Setup
To fetch plant information, you need an OpenAI API key.

Go to OpenAI API

Generate a secret key

Create a .env file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

🛑 Do NOT push .env to GitHub. It contains your private API key.

In your Flask code, load the key using:

python
```bash
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```

🚀 Running the App
1. Clone the repository:
```bash
git clone https://github.com/Efilkucf/FloraAI_medicinal_plant_detection.git
cd FloraAI_medicinal_plant_detection
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
Ensure your requirements.txt contains:
```bash
nginx
Copy
Edit
torch
torchvision
flask
openai
python-dotenv
Pillow
```
3. Start the Flask app:
```bash
python app.py
```
The app will be accessible at:
👉 http://127.0.0.1:5000/

Upload a plant image and get a prediction and detailed description.

🗂️ Project Structure
graphql
```bash
FloraAI_medicinal_plant_detection/
├── app.py                     # Main Flask app
├── train.py                   # Training script (optional)
├── model/
│   └── medicinal_classifier2.pth   # Trained model (use Git LFS or external link)
├── dataset/                   # Local training dataset
├── static/                    # Static files (if any)
├── templates/
│   └── index.html             # HTML frontend
├── .env                       # OpenAI API key (not pushed to GitHub)
├── requirements.txt
├── .gitattributes             # For Git LFS tracking
└── README.md
```
📦 Optional: Upload Model to Google Drive
If .pth is too large, upload to Google Drive and modify your script to download it at runtime.

🙌 Acknowledgements
Kaggle for dataset

PyTorch for model development

OpenAI for plant description generation

Inspired by the intersection of AI and herbal medicine 🌿
