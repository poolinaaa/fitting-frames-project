import torch
from torchvision import models, transforms
from PIL import Image

#ex model path model_path = 'best_model.pth'
def predict_shape(image_path, model_path='models\\best_model.pth'):

    model = models.efficientnet_b4(pretrained=False)
    num_classes = 5
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(model.classifier[1].in_features, num_classes)
    )

    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)
    input_image = input_image.to(device)

    with torch.no_grad():
        output = model(input_image)
        _, predicted_class = torch.max(output, 1)

    class_names = ["heart", "oblong", "oval", "round", "square"]
    predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name

