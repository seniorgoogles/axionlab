import torch.nn as nn
import brevitas.nn as qnn
class Mapper:
    @staticmethod
    def get_layer_by_name(layer_name):
        if hasattr(nn, layer_name):
            return getattr(nn, layer_name)
        elif hasattr(qnn, layer_name):
            return getattr(qnn, layer_name)
        else:
            raise Exception(f"{layer_name} not found in nn or qnn module.")

    @staticmethod
    def has_key(yaml, key):
        try:
            return True if key in yaml else False
        except yaml.YAMLError as exc:
            print(exc)
            return False

    @staticmethod
    def add_model_to_object(model, obj, model_name):
        #@todo change order of inputs
        setattr(obj, model_name, model)
    @staticmethod
    def add_layer_to_object(layer_args, layer, obj, layer_name):
        #@todo change order of inputs
        setattr(obj, layer_name, layer(**layer_args))
