import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
import torchvision.models as models

def map_config_as_attr(config, module, obj, attr_name):

    if issubclass(module, nn.Conv2d):
        in_channels = config[0]
        out_channels = config[1]
        kernel_size = (config[2], config[2])
        stride = (config[3],config[3])
        padding = (config[4],config[4])
        bias = config[5]
        dilation = config[6]
        setattr(obj, attr_name, module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride,padding=padding, bias=bias, dilation=dilation))
    elif issubclass(module, nn.BatchNorm2d):
        num_features = config[0]
        eps = float(config[1])
        momentum = config[2]
        affine = config[3]
        track_running_stats = config[4]
        setattr(obj, attr_name, module(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                       track_running_stats=track_running_stats))
    else:
        setattr(obj, attr_name, module(*config))

def map_config_to_module(config, module):

    if issubclass(module, nn.Conv2d):
        in_channels = config[0]
        out_channels = config[1]
        kernel_size = (config[2], config[2])
        stride = (config[3],config[3])
        padding = (config[4],config[4])
        bias = config[5]
        dilation = config[6]

        return module(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias, dilation=dilation)

    elif issubclass(module, nn.BatchNorm2d):
        num_features = config[0]
        eps = float(config[1])
        momentum = config[2]
        affine = config[3]
        track_running_stats = config[4]
        return module(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                      track_running_stats=track_running_stats)
    else:
        return module(*config)

def get_module(module_class):
    if hasattr(nn, module_class):
        return getattr(nn, module_class)
    elif hasattr(qnn, module_class):
        return getattr(qnn, module_class)
    else:
        raise Exception(f"{module_class} not found in nn or qnn module.")

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
                        module = get_module(module_class)
                        args = module_config[4]
                        block_layers.append(map_config_to_module(args, module))
                    map_config_as_attr(nn.Sequential(*block_layers), nn.Sequential, self, block_name)
            else:
                module_class = layer_config[2]
                module = get_module(module_class)
                name = layer_config[3]
                args = layer_config[4]
                map_config_as_attr(args, module, self, name)


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
            self.load_state_dict(resnet18.state_dict())
    def build(self, config):
        for layer_config in config:
            if isinstance(layer_config, list):
                module_class = layer_config[2]
                module = get_module(module_class)
                args = layer_config[4]
                map_config_as_attr(args, module, self, layer_config[3])
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


