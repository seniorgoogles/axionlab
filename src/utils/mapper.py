import brevitas
import torch.nn as nn
import brevitas.nn as qnn
import src.quantizer as quant

class Mapper:
    
    @staticmethod
    def config_has_key(config_dict, key):
        """
        Checks if the given key exists in the configuration dictionary.
        """
        if key in config_dict.keys():
            return True
        else:
            return False

    @staticmethod 
    def get_quantizer_name_from_conf(config, key):
        """
        Retrieves and removes the quantizer configuration from config using the specified key.
        Returns the corresponding quantizer.
        """
        quantizer_class = config.pop(key, None)
        return Mapper.get_quantizer(quantizer_class)
    
    @staticmethod
    def get_layer_by_name(layer_name):
        """
        Retrieves the layer class by name from nn or qnn modules.
        """
        if layer := getattr(nn, layer_name, None):
            return layer
        if layer := getattr(qnn, layer_name, None):
            return layer
        raise ValueError(f"{layer_name} not found in nn or qnn module.")
        
    @staticmethod
    def get_quantizer(quantizer_class):
        """
        Retrieves the quantizer class from brevitas or custom quant modules.
        """
        if quantizer_class in (None, 'None'):
            return None
        
        for module in [brevitas.quant, brevitas.quant.fixed_point, quant]:
            if hasattr(module, quantizer_class):
                return getattr(module, quantizer_class)
        
        raise ValueError(f"{quantizer_class} not found in brevitas.quant or quant module.")

    @staticmethod
    def add_model_to_object(model, obj, model_name):
        """
        Adds the model as an attribute to the given object.
        """
        setattr(obj, model_name, model)
        
    @staticmethod
    def map_config_as_attr(config, module, obj, attr_name):
        """
        Maps the configuration to a module and sets it as an attribute of the object.
        Handles optional weight and output quantizers.
        """
        layer_config = dict(config)
        
        weight_quant = Mapper.config_has_key(layer_config, 'weight_quant')
        output_quant = Mapper.config_has_key(layer_config, 'output_quant')

        # Pops also the keys if they exist
        weight_quantizer = Mapper.get_quantizer_name_from_conf(layer_config, 'weight_quant') if 'weight_quant' in layer_config else None
        output_quantizer = Mapper.get_quantizer_name_from_conf(layer_config, 'output_quant') if 'output_quant' in layer_config else None
        
        if weight_quant:
            setattr(obj, attr_name, module(**layer_config, weight_quant=weight_quantizer))
        elif output_quant:
            setattr(obj, attr_name, module(**layer_config, output_quant=output_quantizer))
        elif weight_quant and output_quant:
            setattr(obj, attr_name, module(**layer_config, weight_quant=weight_quantizer, output_quant=output_quantizer))
        else:
            setattr(obj, attr_name, module(**layer_config))
