import torch.nn as nn
import torch
from torchvision import models


class CombinedDenseNetEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CombinedDenseNetEfficientNet, self).__init__()
        
        # Load a pretrained DenseNet model
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs_densenet = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs_densenet, num_classes)
        
        # Load a pretrained ResNet model (Replaced EfficientNet with ResNet)
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs_resnet, num_classes)
    
    def forward(self, x):
        # Forward pass through DenseNet
        densenet_output = self.densenet(x)
        
        # Forward pass through ResNet (Previously EfficientNet)
        resnet_output = self.resnet(x)
        
        # Combine the outputs (averaging them)
        combined_output = (densenet_output + resnet_output) / 2
        
        return combined_output
