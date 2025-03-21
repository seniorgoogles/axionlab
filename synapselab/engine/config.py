import yaml 

class Configuration(object):
    
    def __init__(self, config):
        self.keys = []
        self._read_config(config)
                
    def _read_config(self, config_file_path: str):
        """_summary_

        Args:
            config_file_path (_type_): _description_
        """
        if isinstance(config_file_path, str):
            with open(config_file_path, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                    
                    # Iterate over the config and set the attributes
                    for key, value in config.items():
                        self.keys.append(key)
                        setattr(self, key, value)
                                                    
                except yaml.YAMLError as exc:
                    print(exc)
                    
    def _set_dict(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._set_dict(value)
            else:
                setattr(self, key, value)
                
    def set_layer_attribute(self, model_block:str , layer_name:str, key:str, value, force=False):
        """_summary_    
        Args:
            model_block (_type_): e.g. backbone, neck, head
            layer_name (_type_): e.g. dense1, conv2d1
            key (_type_): e.g. features_in, features_out, quantizer
            value (_type_): e.g. 3, 5
        """
        if hasattr(self, model_block):
            model_block = getattr(self, model_block)
            layer_found = False
            for layer in model_block:
                # If layer list has the string layer_name
                if layer_name in layer[3]:
                    layer_found = True
                    if isinstance(layer[4], dict) and key in layer[4]:
                        layer[4][key] = value
                    elif force:
                        layer[4][key] = value
                        # Warning: Key not found in layer, adding it, write to stdout
                        print(f"Warning: Key {key} not found in layer {layer_name}. Adding it.")
                    else:
                        raise ValueError(f"Key {key} not found in layer {layer_name}.")
                    
            if not layer_found:
                raise ValueError(f"Layer {layer_name} not found in model block {model_block}.")
                
        else:
            raise ValueError(f"Model block {model_block} not found in configuration.")

    def remove_layer_attribute(self, model_block:str , layer_name:str, key:str):
        """_summary_    
        Args:
            model_block (_type_): e.g. backbone, neck, head
            layer_name (_type_): e.g. dense1, conv2d1
            key (_type_): e.g. features_in, features_out, quantizer
        """
        if hasattr(self, model_block):
            model_block = getattr(self, model_block)
            layer_found = False
            for layer in model_block:
                # If layer list has the string layer_name
                if layer_name in layer[3]:
                    layer_found = True
                    if isinstance(layer[4], dict) and key in layer[4]:
                        del layer[4][key]
                    else:
                        raise ValueError(f"Key {key} not found in layer {layer_name}.")
                    
            if not layer_found:
                raise ValueError(f"Layer {layer_name} not found in model block {model_block}.")
                
        else:
            raise ValueError(f"Model block {model_block} not found in configuration.")
        
    def export(self, file_path):
        """_summary_

        Args:
            file_path (_type_): _description_
        """
        with open(file_path, 'w') as stream:
            try:
                for key in self.keys:
                    if hasattr(self, key):
                        value = getattr(self, key)
                        # If the value is a list, write each element on a new line
                        if isinstance(value, list):
                            stream.write(f"{key}:\n")
                            for v in value:
                                stream.write(f" - {v}\n")
                        else:   
                            stream.write(f"{key}: {value}\n")
            except yaml.YAMLError as exc:
                print(exc)
