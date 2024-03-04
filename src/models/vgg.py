import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper
class Vgg(nn.Module):
    def __init__(self, config, preload_weights=False):
        super(Vgg, self).__init__()
        self.build(config)

        #if preload_weights:
        #    vgg = models.vgg19(pretrained=True)
        #    self.load_state_dict(vgg.state_dict(), strict=False)

    def build(self, config):
        backbone = config["backbone"]
        for layer_config in backbone:
            if isinstance(layer_config, list):
                module_class = layer_config[2]
                print(module_class)
            elif isinstance(layer_config, dict):
                print(layer_config)

    def forward(self, x):
        x = self.features(x)
        #x = self.av
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x