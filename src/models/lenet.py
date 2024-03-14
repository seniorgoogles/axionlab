import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper


class LeNet5(nn.Module):

    def __init__(self, config, preload_weights=False):
        super(LeNet5, self).__init__()
        self.build(config)

    def build(self, config):
        backbone = config["backbone"]
        for layer_config in backbone:
            module_class = layer_config[2]
            module = Mapper.get_layer_by_name(module_class)
            name = layer_config[3]
            args = layer_config[4]
            Mapper.add_layer_to_object(args, module, self, name)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x