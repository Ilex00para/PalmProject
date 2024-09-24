from torch import nn
import torch
from torchvision.models import efficientnet_b0

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        enb0 = efficientnet_b0(pretrained=True)
        # Replace the first layer with a 1D convolution layer
        self.features = nn.Sequential(
                                      *[self.TwoD_to_OneD(layer) for layer in enb0.features[1:].modules()]
                                      )
        self.input_layer = nn.Sequential(nn.Conv1d(190,32,kernel_size=(3), stride=1, padding=1, bias=False),
                            nn.BatchNorm1d(32),
                            nn.SiLU(inplace=True))
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Linear(1280, 1)

    def TwoD_to_OneD(self, feature):
        for layer in feature.modules():
            if isinstance(layer, nn.Conv2d):
                    return nn.Conv1d(layer.in_channels, layer.out_channels, layer.kernel_size[0], layer.stride[0], layer.padding[0])
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    return nn.AdaptiveAvgPool1d(layer.output_size)
            elif isinstance(layer, nn.BatchNorm2d):
                    return nn.BatchNorm1d(layer.num_features)
            elif isinstance(layer, nn.Sequential):
                for sublayer in layer.modules():
                    if isinstance(sublayer, nn.Conv2d):
                        return nn.Conv1d(sublayer.in_channels, sublayer.out_channels, sublayer.kernel_size[0], sublayer.stride[0], sublayer.padding[0])
                    elif isinstance(sublayer, nn.AdaptiveAvgPool2d):
                        return nn.AdaptiveAvgPool1d(sublayer.output_size)
                    elif isinstance(sublayer, nn.BatchNorm2d):
                        return nn.BatchNorm1d(sublayer.num_features)
                return layer
            else:
                return layer

    def forward(self, x):
        #x = x.reshape(x.shape[0], 40, 190) #just for grid search
        x = torch.transpose(x, 1, 2)
        x = self.input_layer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


if __name__ == '__main__':
    adapted_enb0 = EfficientNet()
    print(adapted_enb0)

    """
    input = torch.rand(190,40)

    input = torch.cat(([input.unsqueeze(1)]*3), dim=1).unsqueeze(0)
    print(input.shape)
    output = adapted_enb0(input)

    print(output)
    """