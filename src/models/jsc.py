import torch
import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper


from torch import Tensor
from torch.autograd.function import Function

class Truncate(Function):
    @staticmethod
    def forward(ctx: Function, input: Tensor, lsb: int):
        lsb = abs(lsb)
        truncated_input = (input * (2 ** lsb)).int().double()
        return truncated_input / (2 ** lsb)

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor):
        return grad_output, None

class Jsc(nn.Module):

    def __init__(self, config, preload_weights=False):
        super(Jsc, self).__init__()
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
    def truncate(self, x, lsb):
        return Truncate.apply(x, lsb)

    def forward(self, x):
        lsb = 3 # to be adjusted to the correct value
        if self.name == "jsc_lite":
            x = self.relu1(self.truncate((self.dense1(x)), lsb))
            x = self.relu2(self.truncate((self.dense2(x)), lsb))
            x = self.truncate(self.dense3(x), lsb)
            x = self.softmax(x)
            return x
        if self.name == "jsc_xl":
            x = self.relu1(self.truncate(self.dense1(x), lsb))
            x = self.relu2(self.truncate(self.dense2(x), lsb))
            x = self.relu3(self.truncate(self.dense3(x), lsb))
            x = self.relu4(self.truncate(self.dense4(x), lsb))
            x = self.truncate(self.dense5(x), lsb)
            x = self.softmax(x)
            return x
        if self.name == "jsc_m_lite_floating_point":
            x = self.relu1(self.dense1(x.type(torch.float)))
            x = self.relu2(self.dense2(x.type(torch.float)))
            x = self.dense3(x.type(torch.float))
            x = self.softmax(x)
            return x
        if self.name == "jsc_xl_floating_point":
            x = self.relu1(self.dense1(x.type(torch.float)))
            x = self.relu2(self.dense2(x.type(torch.float)))
            x = self.relu3(self.dense3(x.type(torch.float)))
            x = self.relu4(self.dense4(x.type(torch.float)))
            x = self.dense5(x.type(torch.float))
            x = self.softmax(x)
            return x
        if self.name == "jsc-2l":
            x = self.relu1(self.truncate(self.dense1(x), lsb))
            x = self.truncate(self.dense2(x), lsb)
            x = self.softmax(x)
            return x
        if self.name == "jsc-5l":
            x = self.relu1(self.truncate(self.dense1(x), lsb))
            x = self.relu2(self.truncate(self.dense2(x), lsb))
            x = self.relu3(self.truncate(self.dense3(x), lsb))
            x = self.relu4(self.truncate(self.dense4(x), lsb))
            x = self.truncate(self.dense5(x), lsb)
            x = self.softmax(x)
            return x
        if self.name == "hdr-5l":
            x = self.relu1(self.truncate(self.dense1(x), lsb))
            x = self.relu2(self.truncate(self.dense2(x), lsb))
            x = self.relu3(self.truncate(self.dense3(x), lsb))
            x = self.relu4(self.truncate(self.dense4(x), lsb))
            x = self.truncate(self.dense5(x), lsb)
            x = self.softmax(x)
            return x
        return x


" Nur eine Idee "


class JscModelFactory:

    @staticmethod
    def create(name, config):
        if name == "jsc":
            return Jsc(config)
        else:
            raise Exception("Model not available")