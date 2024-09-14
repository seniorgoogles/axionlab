import torch.nn as nn
from src.utils.mapper import Mapper

class Jsc(nn.Module):

    def __init__(self, config, preload_weights=False):
        super(Jsc, self).__init__()
        self.num_layers = None
        self.name = None
        self.build(config)

    def build(self, config):
        backbone = config["backbone"]
        self.name = config["name"]

        for layer_config in backbone:
            module_class = layer_config[2]
            module = Mapper.get_layer_by_name(module_class)
            name = layer_config[3]
            args = layer_config[4]
            Mapper.map_config_as_attr(args, module, self, name)

    def forward(self, x):
        if self.name == "jsc_lite" or self.name == "quant_jsc_lite":
            x = self.relu1(self.dense1(x))
            x = self.relu2(self.dense2(x))
            x = self.dense3(x)
            x = self.softmax(x)
        elif self.name == "jsc_xl" or self.name == "quant_jsc_xl":
            x = self.relu1(self.dense1(x))
            x = self.relu2(self.dense2(x))
            x = self.relu3(self.dense3(x))
            x = self.relu4(self.dense4(x))
            x = self.dense5(x)
            x = self.softmax(x)
        elif self.name == "jsc_m_lite" or self.name == "quant_jsc_m_lite":
            x = self.relu1(self.dense1(x))
            x = self.relu2(self.dense2(x))
            x = self.dense3(x)
            x = self.softmax(x)
        elif self.name == "jsc_2l" or self.name == "quant_jsc_2l":
            x = self.relu1(self.dense1(x))
            x = self.dense2(x)
            x = self.softmax(x)
        elif self.name == "jsc_5l" or self.name == "quant_jsc_jsc_5l":
            x = self.relu1(self.dense1(x))
            x = self.relu2(self.dense2(x))
            x = self.relu3(self.dense3(x))
            x = self.relu4(self.dense4(x))
            x = self.dense5(x)
            x = self.softmax(x)
        elif self.name == "jsc_m" or self.name == "quant_jsc_m":
            x = self.relu1(self.dense1(x))
            x = self.relu2(self.dense2(x))
            x = self.relu3(self.dense3(x))
            x = self.relu4(self.dense4(x))
            x = self.dense5(x)
            x = self.softmax(x)
        else:
            raise ValueError(f"Model {self.name} not found")
        return x

