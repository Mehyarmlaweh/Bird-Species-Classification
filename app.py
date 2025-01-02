import gradio as gr
import torch
import pickle
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomErasing
from PIL import Image
import numpy as np
from ultralytics import YOLO 
import io


# Load the YOLO model for bird detection
yolo_model = YOLO('yolov5su.pt')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')  # Ensure loading on CPU
        else:
            return super().find_class(module, name)

# Load your model using the custom unpickler
with open(".\model\model_resultsconvnext_large.pkl", "rb") as file:
    model = CPU_Unpickler(file).load()
    model = model['convnext_large']['model']
model.eval()
# Function to detect bird region
def detect_bird_region(image):
    results = yolo_model(image, verbose=False)
    bird_boxes = results[0].boxes[results[0].boxes.cls == 14] 
    if len(bird_boxes) > 0:
        return bird_boxes[0].xyxy[0].cpu().numpy()  # Coordinates of the first detected bird
    return None

# Preprocessing function for inference
def preprocess_image(image):
    bird_box = detect_bird_region(image)
    if bird_box is not None:
        image = image.crop(bird_box)  # Crop to bird region

    # Apply validation transformations
    val_transform = transforms.Compose([
        transforms.Resize((229, 229)),
        ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return val_transform(image).unsqueeze(0)  # Add batch dimension

# Prediction function
def predict(image):
    # Preprocess the image
    image = preprocess_image(image)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    # Map the predicted class to bird names using bird_folders
    bird_folders = {
        0: "019.Gray_Catbird",
        1: "025.Pelagic_Cormorant",
        2: "026.Bronzed_Cowbird",
        3: "029.American_Crow",
        4: "039.Least_Flycatcher",
        5: "073.Blue_Jay",
        6: "085.Horned_Lark",
        7: "099.Ovenbird",
        8: "104.American_Pipit",
        9: "119.Field_Sparrow",
        10: "127.Savannah_Sparrow",
        11: "129.Song_Sparrow",
        12: "135.Bank_Swallow",
        13: "137.Cliff_Swallow",
        14: "138.Tree_Swallow",
        15: "142.Black_Tern",
        16: "143.Caspian_Tern",
        17: "144.Common_Tern",
        18: "167.Hooded_Warbler",
        19: "176.Prairie_Warbler",
        20: "177.Prothonotary_Warbler",
        21: "179.Tennessee_Warbler",
        22: "182.Yellow_Warbler",
        23: "183.Northern_Waterthrush",
        24: "185.Bohemian_Waxwing",
        25: "186.Cedar_Waxwing",
        26: "188.Pileated_Woodpecker",
        27: "192.Downy_Woodpecker",
        28: "195.Carolina_Wren",
        29: "199.Winter_Wren"
    }
    return bird_folders[predicted_class]  # Return bird name as output

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label"  # Display class label as output
)

# Launch Gradio App
interface.launch()
