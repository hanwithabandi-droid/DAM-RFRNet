import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (Channel Attention) Block
    Recalibrates channel-wise feature responses globally.
    """
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channel = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttentionModule(nn.Module):
    """
    NOVEL BLOCK 1: Spatial Attention Module (SAM)
    Focuses adaptively on spatially heterogeneous haze regions (e.g. background vs foreground).
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_map = self.sigmoid(self.conv(concat))
        return x * spatial_map


class DualAttentionBlock(nn.Module):
    """
    Dual Spatial-Channel Attention Mechanism (DSCA)
    Combines Channel Attention (SE) and Spatial Attention (SAM) for comprehensive feature refinement.
    """
    def __init__(self, channel, reduction=4):
        super(DualAttentionBlock, self).__init__()
        self.se_channel_attention = SEBlock(channel, reduction=reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

    def forward(self, x):
        x_channel = self.se_channel_attention(x)
        x_dual = self.spatial_attention(x_channel)
        return x_dual


class ResidualFeatureRefinementBlock(nn.Module):
    """
    NOVEL BLOCK 2: Multi-Scale Residual Feature Refinement Block (RFRB)
    Recovers high-frequency edge textures blurred during multi-scale upsampling/integration.
    """
    def __init__(self, in_channels=6, mid_channels=6):
        super(ResidualFeatureRefinementBlock, self).__init__()
        self.conv_std = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.conv_dilated = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=True)
        self.fusion = nn.Conv2d(mid_channels * 2, in_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat_std = self.relu(self.conv_std(x))
        feat_dilated = self.relu(self.conv_dilated(x))
        concat = torch.cat([feat_std, feat_dilated], dim=1)
        refined = self.fusion(concat)
        return x + refined  # Residual skip connection


class EnhancedKValueModule(nn.Module):
    """Multi-kernel K-value estimation network."""
    def __init__(self, in_channels=3):
        super(EnhancedKValueModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.conv5.bias, 1.0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        
        cat1 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.conv3(cat1))
        
        cat2 = torch.cat((x2, x3), dim=1)
        x4 = F.relu(self.conv4(cat2))
        
        cat3 = torch.cat((x1, x2, x3, x4), dim=1)
        K = F.relu(self.conv5(cat3))
        
        return K


class DehazeAttentionNet(nn.Module):
    """
    Upgraded Dual-Attention & Multi-Scale Residual Refinement Network (DAM-RFRNet)
    Pipeline:
      RGB Hazy Input -> Preprocessing -> Down-sampling -> Coarse K-Value Estimation
      -> Up-sampling -> Feature Extraction -> Integration with Original RGB Image
      -> NOVEL BLOCK 1: Residual Feature Refinement Block (RFRB)
      -> NOVEL BLOCK 2: Dual Spatial-Channel Attention Mechanism (DSCA)
      -> K-Value Estimation -> Atmospheric Reconstruction -> Final Dehazed Output
    """
    def __init__(self):
        super(DehazeAttentionNet, self).__init__()
        
        # 1. Down-sampled coarse K-value estimation module
        self.compact_k_module = EnhancedKValueModule(in_channels=3)
        
        # 2. Feature extraction block post-upsampling
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # 3. NOVEL BLOCK 1: Residual Feature Refinement Block (RFRB)
        self.refinement_block = ResidualFeatureRefinementBlock(in_channels=6, mid_channels=6)
        
        # 4. NOVEL BLOCK 2: Dual Spatial-Channel Attention Block (DSCA)
        self.dual_attention_block = DualAttentionBlock(channel=6, reduction=2)
        
        # 5. Enhanced K-value module for main path
        self.main_k_module = EnhancedKValueModule(in_channels=6)

    def forward(self, x):
        # 1. Down-sampling path
        x_down = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Coarse K-value estimation
        k_down = self.compact_k_module(x_down)
        
        # Coarse dehazed image at 64x64
        coarse_down = k_down * x_down - k_down + 1.0
        coarse_down = torch.clamp(coarse_down, 0.0, 1.0)
        
        # 2. Up-sampling block
        coarse_up = F.interpolate(coarse_down, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        # 3. Feature Extraction Block
        extracted_features = self.feature_extractor(coarse_up)
        
        # 4. Integration with Original Image Block (Concatenate x + extracted_features -> 6 channels)
        concat_in = torch.cat((x, extracted_features), dim=1)
        
        # 5. Apply NOVEL BLOCK 1: Residual Feature Refinement Block (RFRB)
        refined_features = self.refinement_block(concat_in)
        
        # 6. Apply NOVEL BLOCK 2: Dual Spatial-Channel Attention Block (DSCA)
        attended_features = self.dual_attention_block(refined_features)
        
        # 7. Main K-value estimation & Atmospheric Reconstruction
        K = self.main_k_module(attended_features)
        
        # Final dehazed output J(x) = K(x)*I(x) - K(x) + 1.0
        output = K * x - K + 1.0
        output = torch.clamp(output, 0.0, 1.0)
        
        return output

# Alias DAM_RFRNet to DehazeAttentionNet for exact class name compatibility
DAM_RFRNet = DehazeAttentionNet


class AODNet(nn.Module):
    """Baseline AOD-Net retained for comparative ablation study."""
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.conv5.bias, 1.0)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), dim=1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), dim=1)
        K = F.relu(self.conv5(cat3))
        output = K * x - K + 1.0
        output = torch.clamp(output, 0.0, 1.0)
        return output


if __name__ == '__main__':
    net = DAM_RFRNet()
    print("==================================================")
    print("Novel DAM-RFRNet Architecture Loaded Successfully:")
    print(net)
    print("==================================================")
    
    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_input = torch.sigmoid(dummy_input)
    
    output = net(dummy_input)
    print(f"Input tensor shape:  {dummy_input.shape}")
    print(f"Output tensor shape: {output.shape}")
    
    target = torch.randn(2, 3, 256, 256)
    target = torch.sigmoid(target)
    
    loss = F.mse_loss(output, target)
    loss.backward()
    
    print("Backward pass through RFRB and DSCA blocks successful!")
