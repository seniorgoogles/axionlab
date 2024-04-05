import torch.nn as nn
import brevitas.nn as qnn
import torchvision.models as models
from src.utils.mapper import Mapper


class Jsc(nn.Module):

    def __init__(self, config, preload_weights=False):
        super(Jsc, self).__init__()
        self.build(config)

    def build(self, config):
        backbone = config["backbone"]
        
        name = config["name"]
        
        """
        for layer_config in backbone:
            module_class = layer_config[2]
            module = Mapper.get_layer_by_name(module_class)
            name = layer_config[3]
            args = layer_config[4]
            Mapper.map_config_as_attr(args, module, self, name)
        """

    def forward(self, x):
        
        """
        JSC-2L (Jet Substructure)
        JSC-5L (Jet Substructure)
        JSC-M Lite (Jet Substructure)
        JSC-XL (Jet Substructure)
        
        if jsc_lite:
            pass
        
        elif jsc_xl:
            pass
        """
        return x
    
" Nur eine Idee "
class JscModelFactory:
    
    @staticmethod
    def create(name, config):
        if name == "jsc":
            return Jsc(config)
        else:
            raise Exception("Model not available")