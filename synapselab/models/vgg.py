import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from synapselab.utils.mapper import Mapper
import torch.nn.init as init

class Vgg(nn.Module):
    def __init__(self, config, preload_weights=False):
        super(Vgg, self).__init__()

        backbone = config["backbone"]
        self.build(backbone)

        if preload_weights:
            vgg = models.vgg19(pretrained=True)
            self.load_state_dict(vgg.state_dict(), strict=False)

    def build(self, config):

        for layer_config in config:

            if isinstance(layer_config, list):
                module_class = layer_config[2]
                module_name = layer_config[3]
                config = layer_config[4]
                Mapper.add_layer_to_object(config, Mapper.get_layer_by_name(module_class), self, module_name)

            elif isinstance(layer_config, dict):
                module_name = list(layer_config.keys())[0]
                module_config = layer_config[module_name]
                module = nn.Sequential()

                for layer in module_config:
                    module_class = layer[2]
                    name = layer[3]
                    config = layer[4]
                    module.add_module(name, Mapper.get_layer_by_name(module_class)(**config))

                Mapper.add_model_to_object(module, self, module_name)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x