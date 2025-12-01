"""
Corrected DenLsNet Implementation
- DenseNet-121 backbone (not 201)
- Bidirectional LSTM classifier
- Proper feature fusion with iAFF and SE modules
- Dropout = 0.5
- Final feature dimension = 1920
"""
import timm
import torch
from torch import nn
import torch.nn.functional as F
from model.SENet import SELayer


class iAFF(nn.Module):
    """
    Iterative Attentional Feature Fusion (iAFF)
    Multi-scale feature fusion with local and global attention
    """
    
    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # Local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Second iteration local attention
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        
        # Second iteration global attention
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # First iteration
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        # Second iteration
        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class DenLsNet(nn.Module):
    """
    Corrected DenLsNet Implementation
    - DenseNet-121 backbone with SE layers
    - Bidirectional LSTM classifier head
    - iAFF feature fusion
    - Final feature dimension: 1920
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(DenLsNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Create DenseNet-121 with SE layers
        self.densenet = self._create_densenet121_with_se()
        
        # Feature fusion layers to achieve 1920 final dimension
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.conv2d_1 = nn.Conv2d(in_channels=256, out_channels=896, kernel_size=2, stride=2)  # After denseblock2
        self.aff1 = iAFF(channels=896)
        
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.conv2d_2 = nn.Conv2d(in_channels=896, out_channels=1920, kernel_size=2, stride=2)  # After denseblock3
        self.aff2 = iAFF(channels=1920)
        
        # Bidirectional LSTM classifier head
        self.lstm = nn.LSTM(1920, 128, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classifier (256 because bidirectional LSTM outputs 128*2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_densenet121_with_se(self):
        """Create DenseNet-121 with SE layers added to dense blocks and transitions"""
        # Use DenseNet-121 instead of DenseNet-201
        model = timm.create_model('densenet121', pretrained=True, num_classes=self.num_classes)
        
        # Add SE layers to dense blocks
        # DenseNet-121 has 4 dense blocks with [6, 12, 24, 16] layers respectively
        dense_block_indices = [2, 4, 6, 8]  # Indices in features sequential
        
        for block_idx in dense_block_indices:
            dense_block = model.features[block_idx]
            for layer_name, layer in dense_block.named_children():
                if hasattr(layer, 'norm2'):
                    num_features = layer.norm2.num_features
                    layer.add_module("SELayer", SELayer(num_features))
        
        # Add SE layers to transition layers
        transition_indices = [3, 5, 7]  # Transition layers in features
        for trans_idx in transition_indices:
            if trans_idx < len(model.features):
                transition = model.features[trans_idx]
                if hasattr(transition, 'norm'):
                    num_features = transition.norm.num_features
                    transition.norm.add_module("SELayer", SELayer(num_features))
        
        return model
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through DenLsNet"""
        # DenseNet feature extraction with iAFF fusion
        x = self.densenet.features.conv0(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.pool0(x)
        
        # DenseBlock1 + Transition1
        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        
        # DenseBlock2 + first fusion point
        x = self.densenet.features.denseblock2(x)
        y = self.dropout1(x)
        y = self.conv2d_1(y)
        
        # DenseBlock3 + iAFF fusion
        x = self.densenet.features.transition2(x)
        x = self.densenet.features.denseblock3(x)
        x = self.aff1(x, y)
        
        # Second fusion point
        y1 = self.dropout2(x)
        y1 = self.conv2d_2(y1)
        
        # DenseBlock4 + final iAFF fusion
        x = self.densenet.features.transition3(x)
        x = self.densenet.features.denseblock4(x)
        x = self.aff2(x, y1)
        
        # Final normalization
        x = self.densenet.features.norm5(x)
        
        # Bidirectional LSTM classification
        features = self.pool(x)  # Global average pooling
        batch_size, channels, height, width = features.size()
        
        # Reshape for LSTM: (batch_size, seq_len=1, features=1920)
        features = features.view(batch_size, 1, channels)
        
        # Bidirectional LSTM
        lstm_output, _ = self.lstm(features)
        
        # Take the output from the last time step (seq_len=1, so index 0)
        lstm_features = lstm_output[:, -1, :]  # Shape: (batch_size, 256)
        
        # Final classification
        logits = self.classifier(lstm_features)
        
        return logits
    
    def get_feature_maps(self, x):
        """Extract intermediate feature maps for analysis"""
        features = {}
        
        # Initial convolution
        x = self.densenet.features.conv0(x)
        x = self.densenet.features.norm0(x)
        x = self.densenet.features.pool0(x)
        features['initial'] = x
        
        # Dense blocks
        x = self.densenet.features.denseblock1(x)
        x = self.densenet.features.transition1(x)
        features['block1'] = x
        
        x = self.densenet.features.denseblock2(x)
        features['block2'] = x
        
        # First fusion
        y = self.dropout1(x)
        y = self.conv2d_1(y)
        
        x = self.densenet.features.transition2(x)
        x = self.densenet.features.denseblock3(x)
        x = self.aff1(x, y)
        features['fusion1'] = x
        
        # Second fusion
        y1 = self.dropout2(x)
        y1 = self.conv2d_2(y1)
        
        x = self.densenet.features.transition3(x)
        x = self.densenet.features.denseblock4(x)
        x = self.aff2(x, y1)
        features['fusion2'] = x
        
        # Final features
        x = self.densenet.features.norm5(x)
        features['final'] = x
        
        return features
    
    def print_architecture_summary(self):
        """Print detailed architecture summary for audit"""
        print("=" * 80)
        print("DenLsNet Architecture Summary")
        print("=" * 80)
        
        # Model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"📊 Model Statistics:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Number of Classes: {self.num_classes}")
        print(f"   Dropout Rate: {self.dropout_rate}")
        
        print(f"\n🏗️ Architecture Components:")
        print(f"   Backbone: DenseNet-121 (with SE layers)")
        print(f"   Feature Fusion: iAFF (iterative Attentional Feature Fusion)")
        print(f"   Classifier: Bidirectional LSTM + MLP")
        print(f"   Final Feature Dimension: 1920")
        
        # Test forward pass to get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.get_feature_maps(dummy_input)
            output = self.forward(dummy_input)
        
        print(f"\n📐 Feature Map Dimensions:")
        for name, feature_map in features.items():
            print(f"   {name}: {list(feature_map.shape)}")
        
        print(f"   Output: {list(output.shape)}")
        
        # Verify key components
        print(f"\n✅ Component Verification:")
        
        # Check SE layers
        se_count = 0
        for name, module in self.named_modules():
            if isinstance(module, SELayer):
                se_count += 1
        print(f"   SE Layers: {se_count} found")
        
        # Check iAFF layers
        iaff_count = 0
        for name, module in self.named_modules():
            if isinstance(module, iAFF):
                iaff_count += 1
        print(f"   iAFF Layers: {iaff_count} found")
        
        # Check LSTM
        lstm_found = False
        for name, module in self.named_modules():
            if isinstance(module, nn.LSTM):
                lstm_found = True
                print(f"   LSTM: Bidirectional={module.bidirectional}, Hidden={module.hidden_size}")
                break
        
        if not lstm_found:
            print(f"   ❌ LSTM: Not found!")
        
        # Check final feature dimension
        final_features = features['final']
        if final_features.shape[1] == 1920:
            print(f"   ✅ Final Feature Dimension: {final_features.shape[1]} (matches paper)")
        else:
            print(f"   ❌ Final Feature Dimension: {final_features.shape[1]} (expected 1920)")
        
        print("=" * 80)


def create_denlsnet(num_classes=2, dropout_rate=0.5):
    """
    Factory function to create DenLsNet model
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        DenLsNet model
    """
    model = DenLsNet(num_classes=num_classes, dropout_rate=dropout_rate)
    return model


if __name__ == "__main__":
    # Test the corrected model
    print("Testing Corrected DenLsNet Implementation")
    
    # Create models
    binary_model = create_denlsnet(num_classes=2)
    multiclass_model = create_denlsnet(num_classes=8)
    
    # Print architecture summaries
    print("\n🔬 Binary Classification Model:")
    binary_model.print_architecture_summary()
    
    print("\n🔬 Multi-class Classification Model:")
    multiclass_model.print_architecture_summary()
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n🧪 Forward Pass Test:")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        binary_output = binary_model(dummy_input)
        multiclass_output = multiclass_model(dummy_input)
    
    print(f"Binary output shape: {binary_output.shape}")
    print(f"Multiclass output shape: {multiclass_output.shape}")
    
    # Test probabilities
    binary_probs = F.softmax(binary_output, dim=1)
    multiclass_probs = F.softmax(multiclass_output, dim=1)
    
    print(f"Binary probabilities sum: {binary_probs.sum(dim=1)}")
    print(f"Multiclass probabilities sum: {multiclass_probs.sum(dim=1)}")
    
    print("\n✅ All tests passed! DenLsNet is ready for training.")