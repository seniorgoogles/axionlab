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
        truncated_input = (input * (2 ** lsb)).int().float()
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
        lsb_out = config["lsb_out"]


        if isinstance(lsb_out, int):
            self.lsb_out = [lsb_out] * 5 # 5 is the number of layers. To be adjusted to the correct value
        elif isinstance(self.lsb_out, list):
            self.lsb_out = lsb_out
        else:
            raise TypeError(f"lsb_out must be int or list[int], not {type(lsb_out)}")

        for layer_config in backbone:
            module_class = layer_config[2]
            module = Mapper.get_layer_by_name(module_class)
            name = layer_config[3]
            args = layer_config[4]
            Mapper.map_config_as_attr(args, module, self, name)

        print("Model built")
    def truncate(self, x, lsb):
        return Truncate.apply(x, lsb)

    def forward(self, x):

        if self.name == "jsc_lite":
            x = self.relu1(self.truncate((self.dense1(x)), self.lsb_out[0]))
            x = self.relu2(self.truncate((self.dense2(x)), self.lsb_out[1]))
            x = self.truncate(self.dense3(x), self.lsb_out[2])
            x = self.softmax(x)
        if self.name == "jsc_xl":
            x = self.relu1(self.truncate(self.dense1(x), self.lsb_out[0]))
            x = self.relu2(self.truncate(self.dense2(x), self.lsb_out[1]))
            x = self.relu3(self.truncate(self.dense3(x), self.lsb_out[2]))
            x = self.relu4(self.truncate(self.dense4(x), self.lsb_out[3]))
            x = self.truncate(self.dense5(x), self.lsb_out[4])
            x = self.softmax(x)
        if self.name == "jsc_m_lite_floating_point":
            x = self.relu1(self.dense1(x.type(torch.float)))
            x = self.relu2(self.dense2(x.type(torch.float)))
            x = self.dense3(x.type(torch.float))
            x = self.softmax(x)
        if self.name == "jsc_xl_floating_point":
            x = self.relu1(self.dense1(x.type(torch.float)))
            x = self.relu2(self.dense2(x.type(torch.float)))
            x = self.relu3(self.dense3(x.type(torch.float)))
            x = self.relu4(self.dense4(x.type(torch.float)))
            x = self.dense5(x.type(torch.float))
            x = self.softmax(x)
        if self.name == "jsc-2l":
            x = self.relu1(self.truncate(self.dense1(x), self.lsb_out[2]))
            x = self.truncate(self.dense2(x), self.lsb_out[1])
            x = self.softmax(x)
        if self.name == "jsc-5l":
            x = self.relu1(self.truncate(self.dense1(x), self.lsb_out[0]))
            x = self.relu2(self.truncate(self.dense2(x), self.lsb_out[1]))
            x = self.relu3(self.truncate(self.dense3(x), self.lsb_out[2]))
            x = self.relu4(self.truncate(self.dense4(x), self.lsb_out[3]))
            x = self.truncate(self.dense5(x), self.lsb_out[4])
            x = self.softmax(x)
        if self.name == "hdr-5l":
            x = self.relu1(self.truncate(self.dense1(x), self.lsb_out[0]))
            x = self.relu2(self.truncate(self.dense2(x), self.lsb_out[1]))
            x = self.relu3(self.truncate(self.dense3(x), self.lsb_out[2]))
            x = self.relu4(self.truncate(self.dense4(x), self.lsb_out[3]))
            x = self.truncate(self.dense5(x), self.lsb_out[4])
            x = self.softmax(x)
        return x

