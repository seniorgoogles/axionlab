import torch
import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper


class ContinuousflowMnist(nn.Module):

    def __init__(self, config, weights_path=None):
        super(ContinuousflowMnist, self).__init__()
        self.num_layers = None
        self.name = None
        self.build(config)
        
        if weights_path:
            self.load_state_dict(torch.load(weights_path))
            
        self.bn1 = nn.BatchNorm2d(1)         
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(4)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(64)
        
    def build(self, config):
        self.name = config["name"]
        backbone = config["backbone"]

        for layer_config in backbone:
            module_class = layer_config[2]
            module = Mapper.get_layer_by_name(module_class)
            name = layer_config[3]
            args = layer_config[4]
            Mapper.map_config_as_attr(args, module, self, name)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise1(x)))
        x = self.relu2(self.bn2(self.pointwise1(x)))
        x = self.relu3(self.bn3(self.depthwise2(x)))
        x = self.relu4(self.bn4(self.pointwise2(x)))
        x = self.relu5(self.bn5(self.depthwise3(x)))
        x = self.relu6(self.bn6(self.pointwise3(x)))
        x = self.dense(self.flatten(x))
        x = self.softmax(x)

        return x
