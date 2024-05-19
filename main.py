import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List
import os
from PIL import Image



class ImprovedTinyVGG(nn.Module):
    """
    Improved version of the TinyVGG model architecture
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1          = nn.Sequential(
            nn.Conv2d(in_channels  =  input_shape, 
                                      out_channels = 32, 
                                      kernel_size  = 3, 
                                      padding      = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels  = 32, 
                      out_channels = 64, 
                      kernel_size  = 3, 
                      padding      = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels  = 64, 
                      out_channels = 128, 
                      kernel_size  = 3, 
                      padding      = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels    = 128, 
                      out_channels   = 128, 
                      kernel_size    = 3, 
                      padding        = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, 
                         stride      = 2)
        )
        self.fc_layers               = nn.Sequential(
            nn.Linear(in_features    = 32768, 
                      out_features   = 512),  # Adjusted the input size to match the flattened size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features    = 512, 
                      out_features   = output_shape)
)


    def forward(self, x):
        x = self.conv_block_1(x)
#         print(x.shape)
        x = self.conv_block_2(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
#         print(x.shape)
        x = self.fc_layers(x)
#         print(x.shape)
        return x



class ImagePredictor:
    def __init__(self, model, class_names=None, device=torch.device('cpu')):
        self.model = model
        self.class_names = class_names
        self.device = device

    def pred_and_plot_image(self, uploaded_image: Image.Image, custom_image_transform=None):
        """Makes a prediction on an uploaded image with a trained model and plots the image and prediction."""
        # Convert PIL image to tensor
        image_tensor = transforms.ToTensor()(uploaded_image)

        # Apply custom transformation if provided
        if custom_image_transform:
            image_tensor = custom_image_transform(image_tensor)

        # Make sure the model is on target device
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            # Add an extra dimension to the image (this is the batch dimension, e.g. our model will predict on batches of 1x image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Make a prediction on the image with an extra dimension
            image_pred = self.model(image_tensor)

        # Convert logits -> prediction probabilities
        image_pred_probs = torch.softmax(image_pred, dim=1)

        # Convert prediction probabilities -> prediction labels
        image_pred_label = torch.argmax(image_pred_probs, dim=1)

        # Plot the image alongside the prediction and prediction probabilities
        plt.imshow(uploaded_image)
        if self.class_names:
            title = f"Pred: {self.class_names[image_pred_label.cpu().item()]} | Prob: {image_pred_probs.max().cpu().item():.3f}"
        else:
            title = f"Pred: {image_pred_label.item()} | Prob: {image_pred_probs.max().cpu().item():.3f}"
        plt.title(title)
        plt.axis(False)
        plt.show()

        if self.class_names:
            return self.class_names[image_pred_label.cpu().item()]
        else:
            return image_pred_label.item()



