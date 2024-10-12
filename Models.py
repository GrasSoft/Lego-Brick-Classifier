import torch.nn as nn


# Define the modified ResNet50 model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        # Define the new fully connected layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.batch_norm = nn.BatchNorm1d(num_classes)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)  # Pass through ResNet18
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x
        
# Define the modified ResNet34 model
class CustomResNet34(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet34, self).__init__()
        # Load the pretrained ResNet34 model
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        # Define the new fully connected layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.batch_norm = nn.BatchNorm1d(num_classes)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)  # Pass through ResNet50
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Define the modified ResNet50 model
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        # Load the pretrained ResNet50 model
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        # Define the new fully connected layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.batch_norm = nn.BatchNorm1d(num_classes)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet(x)  # Pass through ResNet50
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x