from utils import load_model
from torchvision.models import resnet18
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import json

filename = 'resnet18_weights_best_acc.tar' # pre-trained model path
use_gpu = False  # load weights on the gpu
model = resnet18(num_classes=1081) # 1081 classes in Pl@ntNet-300K

load_model(model, filename=filename, use_gpu=use_gpu)

model.eval()

# Load the JSON mappings
with open("class_idx_to_species_id.json", "r") as f:
    class_idx_to_species_id = json.load(f)

with open("plantnet300K_species_id_2_name.json", "r") as f:
    species_id_to_name = json.load(f)

image_path = "plant.jpg"
image = Image.open(image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize the image to at least 256x256
    transforms.CenterCrop(224),  # Crop it to 224x224 as expected by ResNet
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Apply preprocessing
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# If using GPU, move the input and model to GPU
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)
model.to(device)

# Run the model and get predictions
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(input_batch)

# Get the predicted class index (the class with the highest score)
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()

species_id = class_idx_to_species_id.get(str(predicted_class))
species_name = species_id_to_name.get(str(species_id), "Unknown Species")


draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # You can specify a custom font here if you have one

# Set the position for the text, e.g., top-left corner
text_position = (10, 10)
text_color = (255, 0, 0)  # Red color for text
draw.text(text_position, f"Prediction: {species_name}", fill=text_color, font=font)

# Save the image with detections
output_image_path = "plant_with_detections.jpg"
image.save(output_image_path)
print(f"Image saved with detections at: {output_image_path}")