import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper

class BasicBlock(nn.Module):

    def __init__(self, config):
        super(BasicBlock, self).__init__()
        self.build(config)

    def build(self, config):
        layer_blocks = config["BasicBlock"]
        for layer_config in layer_blocks:

            # If config has downsample configuration
            if isinstance(layer_config, dict):
                for block_name, block_config in layer_config.items():
                    block_layers = []
                    for module_config in block_config:
                        module_class = module_config[2]
                        module = Mapper.get_module(module_class)
                        args = module_config[4]
                        block_layers.append(Mapper.map_config_to_module(args, module))
                    Mapper.map_config_as_attr(nn.Sequential(*block_layers), nn.Sequential, self, block_name)
            else:
                module_class = layer_config[2]
                module = get_module(module_class)
                name = layer_config[3]
                args = layer_config[4]
                Mapper.map_config_as_attr(args, module, self, name)


    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if hasattr(self, "downsample"):
            identity = self.downsample(identity)

        out += identity
        self.relu(out)
        return out

class ResNet(nn.Sequential):
    def __init__(self, config, preload_weights=False):
        super(ResNet, self).__init__()
        self.build(config["backbone"])

        if preload_weights:
            resnet18 = models.resnet18(pretrained=True)
            self.load_state_dict(resnet18.state_dict(), strict=False)
    def build(self, config):
        for layer_config in config:
            if isinstance(layer_config, list):
                module_class = layer_config[2]
                module = get_module(module_class)
                args = layer_config[4]
                Mapper.map_config_as_attr(args, module, self, layer_config[3])
            elif isinstance(layer_config, dict):
                for block_name, block_config in layer_config.items():
                    block_layers = []
                    for module_config in block_config:
                        block_layers.append(BasicBlock(module_config))
                    setattr(self, block_name, nn.Sequential(*block_layers))

    def preload(self):
        pass

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


