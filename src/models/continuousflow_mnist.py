import torch
import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper


class ContinuousflowMnist(nn.Module):

    def __init__(self, config, weights_path=None):
        super(ContinuousflowMnist, self).__init__()
        self.build(config)
        
        if weights_path:
            self.load_state_dict(torch.load(weights_path))
            
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
        x = self.relu1(self.depthwise1(x))
        x = self.relu2(self.pointwise1(x))
        x = self.relu3(self.depthwise2(x))
        x = self.relu4(self.pointwise2(x))
        x = self.relu5(self.depthwise3(x))
        x = self.relu6(self.pointwise3(x))
        x = self.dense(self.flatten(x))
        x = self.softmax(x)

        return x
