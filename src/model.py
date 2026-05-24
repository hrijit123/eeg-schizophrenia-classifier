import torch
import torch.nn as nn
import torch.nn.functional as F

class OriginalEEG2D(nn.Module):
    """
    PyTorch translation of the original Conv2D Keras model from the Kaggle notebook.
    Takes flat EEG trial vectors, reshapes them to (Batch, 1, Channels, Timesteps),
    applies 2D convolutions, max pooling, dropout, and a dense classifier.
    """
    def __init__(self, num_channels=70, timesteps=576):
        super(OriginalEEG2D, self).__init__()
        self.num_channels = num_channels
        self.timesteps = timesteps
        
        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 20))
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 15))
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 13, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3))
        
        self.dropout = nn.Dropout(0.2)
        
        # Determine flattening dimension dynamically
        self.flatten_dim = self._get_flatten_dim()
        
        self.fc1 = nn.Linear(self.flatten_dim, 317)
        self.fc2 = nn.Linear(317, 1)  # Outputs logits for BCEWithLogitsLoss

    def _get_flatten_dim(self):
        """Helper to compute the size of the flattened features after conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_channels, self.timesteps)
            x = F.tanh(self.conv1(x))
            x = self.pool1(x)
            x = F.tanh(self.conv2(x))
            x = self.pool2(x)
            return x.numel()

    def forward(self, x):
        # Input shape: (Batch, num_channels * timesteps)
        # Reshape to 4D: (Batch, Channels=1, Height=num_channels, Width=timesteps)
        x = x.view(-1, 1, self.num_channels, self.timesteps)
        
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EEG1D(nn.Module):
    """
    1D CNN architecture specifically suited for time series classification.
    Treats channels (electrodes) as input channels and performs 1D convolutions over time.
    Shape: (Batch, num_channels, timesteps)
    """
    def __init__(self, num_channels=70, timesteps=576):
        super(EEG1D, self).__init__()
        self.num_channels = num_channels
        self.timesteps = timesteps
        
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.dropout = nn.Dropout(0.2)
        
        # Calculate flatten dim
        self.flatten_dim = self._get_flatten_dim()
        
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def _get_flatten_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, self.num_channels, self.timesteps)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            return x.numel()

    def forward(self, x):
        # Input shape: (Batch, num_channels * timesteps)
        # Reshape to (Batch, num_channels, timesteps)
        x = x.view(-1, self.num_channels, self.timesteps)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EEGNet(nn.Module):
    """
    Standard EEGNet architecture (Lawhern et al., 2018).
    Compact CNN design for EEG signals using temporal, depthwise, and separable 2D convolutions.
    """
    def __init__(self, num_channels=70, timesteps=576, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        self.num_channels = num_channels
        self.timesteps = timesteps
        
        # Step 1: Temporal Conv
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Step 2: Spatial (Depthwise) Conv
        self.conv2 = nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Step 3: Separable Conv
        self.conv3 = nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv3_pointwise = nn.Conv2d(F2, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Flatten and Classify
        self.flatten_dim = self._get_flatten_dim()
        self.classifier = nn.Linear(self.flatten_dim, 1)

    def _get_flatten_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_channels, self.timesteps)
            x = self.bn1(self.conv1(x))
            x = self.bn2(self.conv2(x))
            x = self.dropout2(self.pool2(F.elu(x)))
            # Separable conv
            x = self.conv3_pointwise(self.conv3(x))
            x = self.bn3(x)
            x = self.dropout3(self.pool3(F.elu(x)))
            return x.numel()

    def forward(self, x):
        # Input shape: (Batch, num_channels * timesteps)
        x = x.view(-1, 1, self.num_channels, self.timesteps)
        
        # Block 1
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.conv3_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name="original", num_channels=70, timesteps=576):
    """Model factory function."""
    model_name = model_name.lower().strip()
    if model_name == "original" or model_name == "cnn2d":
        return OriginalEEG2D(num_channels=num_channels, timesteps=timesteps)
    elif model_name == "eeg1d" or model_name == "cnn1d":
        return EEG1D(num_channels=num_channels, timesteps=timesteps)
    elif model_name == "eegnet":
        return EEGNet(num_channels=num_channels, timesteps=timesteps)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'original', 'eeg1d', 'eegnet'")
