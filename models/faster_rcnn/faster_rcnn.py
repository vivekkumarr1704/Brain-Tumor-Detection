import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class FasterRCNNClassifier:
    def __init__(self, num_classes=2):
        # Load pretrained model backbone
        model = models.resnet50(weights="IMAGENET1K_V1")

        # Replace classifier head
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, img):
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img)
            probs = torch.softmax(output, dim=1).numpy()[0]
        return probs
