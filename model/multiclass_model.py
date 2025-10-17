"""
Multi-class extension of DenLsNet for 8-class BreakHis classification
Extends the existing binary model to support multi-class classification
"""
import timm
import torch
from torch import nn
import torch.nn.functional as F

import config_multiclass as config
from model.SENet import SELayer
from model.model import iAFF


class MultiClassDenLsNet(nn.Module):
    """
    Multi-class version of DenLsNet (DenLsNet-MC)
    Extends binary classification to 8-class BreakHis classification
    """
    
    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.5):
        super(MultiClassDenLsNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Base DenseNet with SE layers (same as original)
        self.densenet = self._create_densenet_with_se()
        
        # Feature fusion layers (same as original)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.conv2d_1 = nn.Conv2d(in_channels=512, out_channels=1792, kernel_size=2, stride=2)
        self.aff1 = iAFF(channels=1792)
        
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.conv2d_2 = nn.Conv2d(in_channels=1792, out_channels=1920, kernel_size=2, stride=2)
        self.aff2 = iAFF(channels=1920)
        
        # LSTM classifier (modified for multi-class)
        self.lstm = nn.LSTM(1920, 256, batch_first=True, dropout=dropout_rate)  # Increased hidden size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multi-class classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)  # Softmax output for multi-class
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_densenet_with_se(self):
        """Create DenseNet with SE layers (same as original implementation)"""
        model = timm.create_model('densenet201', pretrained=True, num_classes=self.num_classes)
        
        # Add SE layers to dense blocks
        dense_layers = []
        for name, module in model.named_modules():
            if isinstance(module, timm.models.densenet.DenseBlock):
                dense_layers.append(module)
        
        label = [3, 5, 7, 9]
        for i, dense_layer in enumerate(dense_layers):
            for j in dense_layer:
                num_features = model.features[label[i]][f'{j}'].norm2.num_features
                model.features[label[i]][f'{j}'].add_module("SELayer", SELayer(num_features))
        
        # Add SE layers to transitions
        transition_layers = [
            (model.features.transition1.norm, "transition1"),
            (model.features.transition2.norm, "transition2"), 
            (model.features.transition3.norm, "transition3")
        ]
        
        for transition_norm, name in transition_layers:
            se_layer = SELayer(transition_norm.num_features)
            transition_norm.add_module("SELayer", se_layer)
        
        return model
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # DenseNet feature extraction with iAFF fusion (same as original)
        x = self.densenet.features.conv0(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.pool0(x)
        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        x = self.densenet.features.denseblock2(x)
        
        # First fusion point
        y = self.conv2d_1(x)
        x = self.densenet.features.transition2(x)
        x = self.densenet.features.denseblock3(x)
        x = self.aff1(x, y)
        
        # Second fusion point
        y1 = self.conv2d_2(x)
        x = self.densenet.features.transition3(x)
        x = self.densenet.features.denseblock4(x)
        x = self.aff2(x, y1)
        x = self.densenet.features.norm5(x)
        
        # LSTM classification
        features = self.pool(x)
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, width, channels)
        
        lstm_output, _ = self.lstm(features)
        x = lstm_output[:, -1, :]  # Take last time step
        
        # Multi-class classification
        logits = self.classifier(x)
        
        return logits
    
    def get_features(self, x):
        """Extract features for interpretability analysis"""
        # Same forward pass but return intermediate features
        features = {}
        
        x = self.densenet.features.conv0(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.pool0(x)
        features['block1_input'] = x
        
        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        features['block2_input'] = x
        
        x = self.densenet.features.denseblock2(x)
        y = self.conv2d_1(x)
        x = self.densenet.features.transition2(x)
        x = self.densenet.features.denseblock3(x)
        x = self.aff1(x, y)
        features['block3_output'] = x
        
        y1 = self.conv2d_2(x)
        x = self.densenet.features.transition3(x)
        x = self.densenet.features.denseblock4(x)
        x = self.aff2(x, y1)
        x = self.densenet.features.norm5(x)
        features['final_features'] = x
        
        # LSTM features
        pooled_features = self.pool(x)
        batch_size, channels, height, width = pooled_features.size()
        lstm_input = pooled_features.view(batch_size, width, channels)
        
        lstm_output, _ = self.lstm(lstm_input)
        lstm_features = lstm_output[:, -1, :]
        features['lstm_features'] = lstm_features
        
        # Final logits
        logits = self.classifier(lstm_features)
        features['logits'] = logits
        
        return features


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-class classification
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross-entropy for multi-class classification
    """
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / inputs.size(1)
        
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        return loss.mean()


def create_multiclass_model(
    num_classes: int = 8,
    dropout_rate: float = 0.5,
    pretrained: bool = True
) -> MultiClassDenLsNet:
    """
    Factory function to create multi-class DenLsNet model
    
    Args:
        num_classes: Number of classes (default: 8 for BreakHis)
        dropout_rate: Dropout rate for regularization
        pretrained: Whether to use pretrained DenseNet weights
        
    Returns:
        MultiClassDenLsNet model
    """
    model = MultiClassDenLsNet(num_classes=num_classes, dropout_rate=dropout_rate)
    
    if pretrained:
        print(f"Created multi-class DenLsNet with {num_classes} classes (pretrained)")
    else:
        print(f"Created multi-class DenLsNet with {num_classes} classes (random init)")
    
    return model


def get_loss_function(loss_type: str = 'crossentropy', **kwargs):
    """
    Get appropriate loss function for multi-class classification
    
    Args:
        loss_type: Type of loss ('crossentropy', 'focal', 'label_smoothing')
        **kwargs: Additional arguments for loss functions
        
    Returns:
        Loss function
    """
    if loss_type == 'crossentropy':
        class_weights = kwargs.get('class_weights', None)
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 1.0)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the multi-class model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_multiclass_model(num_classes=8)
    model.to(device)
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Test loss functions
    targets = torch.randint(0, 8, (batch_size,)).to(device)
    
    # Cross-entropy loss
    ce_loss = get_loss_function('crossentropy')
    ce_value = ce_loss(outputs, targets)
    print(f"CrossEntropy Loss: {ce_value.item():.4f}")
    
    # Focal loss
    focal_loss = get_loss_function('focal', alpha=1.0, gamma=2.0)
    focal_value = focal_loss(outputs, targets)
    print(f"Focal Loss: {focal_value.item():.4f}")
    
    # Label smoothing loss
    ls_loss = get_loss_function('label_smoothing', smoothing=0.1)
    ls_value = ls_loss(outputs, targets)
    print(f"Label Smoothing Loss: {ls_value.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")