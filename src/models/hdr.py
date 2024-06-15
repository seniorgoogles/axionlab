import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper


class Hdr(nn.Module):

    def __init__(self, config, preload_weights=False):
        super(Hdr, self).__init__()
        self.build(config)

    def build(self, config):
        backbone = config["backbone"]

        for layer_config in backbone:
            module_class = layer_config[2]
            module = Mapper.get_layer_by_name(module_class)
            name = layer_config[3]
            args = layer_config[4]
            Mapper.map_config_as_attr(args, module, self, name)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.relu4(self.dense4(x))
        x = self.dense5(x)
        x = self.softmax(x)

        return x
