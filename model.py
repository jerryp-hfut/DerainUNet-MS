import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class DoubleConv(nn.Cell):
    """双卷积块：卷积 -> BN -> ReLU -> 卷积 -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=True, weight_init='xavier_uniform'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=True, weight_init='xavier_uniform'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def construct(self, x):
        return self.double_conv(x)

class UNet(nn.Cell):
    """U-Net 网络实现，不使用 for 循环显式指定每个模块"""
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 64)    # 输入 3，输出 64
        self.down2 = DoubleConv(64, 128)            # 输入 64，输出 128
        self.down3 = DoubleConv(128, 256)           # 输入 128，输出 256
        self.down4 = DoubleConv(256, 512)           # 输入 256，输出 512
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv(512, 1024)     # 输入 512，输出 1024
        
        self.up4_trans = nn.Conv2dTranspose(1024, 512, kernel_size=2, stride=2, has_bias=True)
        self.up4_conv = DoubleConv(1024, 512)       # 输入 512+512=1024，输出 512
        self.up3_trans = nn.Conv2dTranspose(512, 256, kernel_size=2, stride=2, has_bias=True)
        self.up3_conv = DoubleConv(512, 256)        # 输入 256+256=512，输出 256
        self.up2_trans = nn.Conv2dTranspose(256, 128, kernel_size=2, stride=2, has_bias=True)
        self.up2_conv = DoubleConv(256, 128)        # 输入 128+128=256，输出 128
        self.up1_trans = nn.Conv2dTranspose(128, 64, kernel_size=2, stride=2, has_bias=True)
        self.up1_conv = DoubleConv(128, 64)         # 输入 64+64=128，输出 64
        
        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, has_bias=True)

    def construct(self, x):
        skip1 = self.down1(x)
        x = self.pool(skip1)
        skip2 = self.down2(x)
        x = self.pool(skip2)
        skip3 = self.down3(x)
        x = self.pool(skip3)
        skip4 = self.down4(x)
        x = self.pool(skip4)
        
        x = self.bottleneck(x)
        
        x = self.up4_trans(x)
        if x.shape[2:] != skip4.shape[2:]:
            x = ops.interpolate(x, size=skip4.shape[2:], mode='bilinear', align_corners=True)
        x = ops.cat([skip4, x], axis=1)
        x = self.up4_conv(x)
        
        x = self.up3_trans(x)
        if x.shape[2:] != skip3.shape[2:]:
            x = ops.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        x = ops.cat([skip3, x], axis=1)
        x = self.up3_conv(x)
        
        x = self.up2_trans(x)
        if x.shape[2:] != skip2.shape[2:]:
            x = ops.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        x = ops.cat([skip2, x], axis=1)
        x = self.up2_conv(x)
        
        x = self.up1_trans(x)
        if x.shape[2:] != skip1.shape[2:]:
            x = ops.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        x = ops.cat([skip1, x], axis=1)
        x = self.up1_conv(x)
        
        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    # 创建模型实例
    model = UNet(in_channels=3, out_channels=3)
    
    # 创建随机输入张量
    input_tensor = mindspore.Tensor(ops.randn(1, 3, 256, 256))
    
    # 前向传播
    output_tensor = model(input_tensor)
    
    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")