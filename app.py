from flask import Flask, request, render_template
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import Dataset
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Flask app setup
app = Flask(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset loader
class MedicinalPlantDataset(Dataset):
    def __init__(self, root_dirs, prefixes, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = set()

        for root_dir, prefix in zip(root_dirs, prefixes):
            for cls in sorted(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, cls)
                if os.path.isdir(class_path):
                    class_name = f"{prefix}_{cls}"
                    self.classes.add(class_name)
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                            self.image_paths.append(os.path.join(class_path, img_name))
                            self.labels.append(class_name)

        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset paths
leaf_dataset_path = r"C:\Users\Keerthana Reddy\Downloads\archive (1)\Indian Medicinal Leaves Image Datasets\Medicinal Leaf dataset"
plant_dataset_path = r"C:\Users\Keerthana Reddy\Downloads\archive (1)\Indian Medicinal Leaves Image Datasets\Medicinal plant dataset"

# Load dataset
dataset = MedicinalPlantDataset([leaf_dataset_path, plant_dataset_path], ["Leaf", "Plant"], transform)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# Load model
model = models.efficientnet_v2_l(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(idx_to_class))
model_path = r"C:\Users\Keerthana Reddy\Downloads\medicinal_classifier1.pth"
state_dict = torch.load(model_path, map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Azure GitHub API setup
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1"
token = "use-your-own-api-key"


client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def get_medicinal_uses_via_ai(plant_name):
    prompt = f"What are the medicinal uses of {plant_name}? Respond in 4-5 lines."
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a medicinal plant expert."),
                UserMessage(content=prompt),
            ],
            temperature=0.7,
            top_p=1,
            model=model_name
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching AI response: {e}"

def predict_and_verify(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    predicted_class = idx_to_class[pred.item()]
    confidence_score = confidence.item() * 100
    ai_response = get_medicinal_uses_via_ai(predicted_class)
    return predicted_class, confidence_score, ai_response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files['file']
        image_path = os.path.join("static", image.filename)
        image.save(image_path)
        predicted_class, confidence_score, ai_response = predict_and_verify(image_path)
        return render_template("index.html",
                               predicted_class=predicted_class,
                               confidence_score=confidence_score,
                               ai_response=ai_response,
                               image_url=image_path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)