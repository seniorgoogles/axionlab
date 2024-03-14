import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper
class BasicBlock(nn.Module):

    def __init__(self, config):
        super(BasicBlock, self).__init__()
        self.build(config)

    def build(self, config):

        for layer_config in config:
            # Adding layer
            if isinstance(layer_config, list):
                module_class = layer_config[2]
                module_name = layer_config[3]
                config = layer_config[4]
                Mapper.add_layer_to_object(config, Mapper.get_layer_by_name(module_class), self, module_name)

            # Adding downsampling layer
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
        identity = x

        # If it is ResNet50
        if hasattr(self, "conv3"):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

            if hasattr(self, "downsample"):
                identity = self.downsample(identity)

            out += identity
            self.relu(out)
        # Else it is ResNet18
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            if hasattr(self, "downsample"):
                identity = self.downsample(identity)

            out += identity
            self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, config, preload_weights=False):
        super(ResNet, self).__init__()
        self.build(config["backbone"])

        # Load weights
        if preload_weights:
            resnet = None
            if config["name"] == "resnet18":
                resnet = models.resnet18(pretrained=True)
            elif config["name"] == "resnet50":
                resnet = models.resnet50(pretrained=True)
            else:
                raise ValueError(f"Invalid {config['name']} model name")

            self.load_state_dict(resnet.state_dict(), strict=True)

    def build(self, config):
        for layer_config in config:
            if isinstance(layer_config, list):
                layer_class = layer_config[2]
                layer_name = layer_config[3]
                layer_args = layer_config[4]

                Mapper.add_layer_to_object(layer_args, Mapper.get_layer_by_name(layer_class), self, layer_name)

            elif isinstance(layer_config, dict):
                layer_name = list(layer_config.keys())[0]
                layer_config_basic_blocks = layer_config[layer_name]

                basic_blocks = []
                for basicblock in layer_config_basic_blocks:
                    basic_block = None

                    if "BasicBlock" in basicblock:
                        basic_block_args = basicblock["BasicBlock"]
                        basic_block = BasicBlock(basic_block_args)

                    elif "Bottleneck" in basicblock:
                        basic_block_args = basicblock["Bottleneck"]

                        # Rename class
                        setattr(BasicBlock, '__name__', 'Bottleneck')
                        basic_block = BasicBlock(basic_block_args)

                    basic_blocks.append(basic_block)

                Mapper.add_model_to_object(nn.Sequential(*basic_blocks), self, layer_name)

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


