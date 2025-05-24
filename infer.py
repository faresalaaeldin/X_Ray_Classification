from torchvision import transforms, models
import torch
import torch.nn as nn
import pickle
from PIL import Image  # Fixed import

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture   
model = models.resnet50(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load trained weights
model.load_state_dict(torch.load(r"D:\Projects\github\X_Ray_Classification\models\best_model.pth", map_location=device))
model.eval()
model.to(device)

# Load label encoder
with open(r"D:\Projects\github\X_Ray_Classification\data\lbl_encoder.pkl", "rb") as f:
    lbl_encoder = pickle.load(f)

# Inference function
def infer(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to match model input size
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        argmax = torch.argmax(output, dim=1).item()
        predicted_class = lbl_encoder.inverse_transform([argmax])[0]
    return predicted_class

# Run inference
result = infer(r"D:\Projects\github\X_Ray_Classification\data\chest_xray\chest_xray\test\PNEUMONIA\person30_virus_69.jpeg")
print(result)
