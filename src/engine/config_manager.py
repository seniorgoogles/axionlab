import yaml 

class ConfigurationManager(object):
    """_summary_

    Args:
        object (_type_): _description_
    """
    def __init__(self, config):
        """ Initializes the ConfigurationManager with a config file.

        Args:
            config_file_path (str): config file path
        """
        
        if isinstance(config, str):
            self.config_file_path = config
            with open(config, 'r') as stream:
                try:
                    self.config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    self.config = None
                    print(exc)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("Invalid config type. Must be a file path or a dictionary.")
        
    def read(self, file_path):
        """_summary_

        Args:
            file_path (_type_): _description_
        """
        with open(file_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    def write(self, file_path):
        """_summary_

        Args:
            file_path (_type_): _description_
        """
        with open(file_path, 'w') as stream:
            try:
                for key in self.config.keys():
                    # If the value is a list, write each element on a new line
                    if isinstance(self.config[key], list):
                        stream.write(f"{key}:\n")
                        for value in self.config[key]:
                            stream.write(f" - {value}\n")
                    else:   
                        stream.write(f"{key}: {self.config[key]}\n")
                        
            except yaml.YAMLError as exc:
                print(exc)
                
    def generate_model_py(self, file_path):
        """_summary_

        Args:
            file_path (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method is not implemented yet.")
    
    def set_layer_param(self, model_part, layer_name, key, value):
        """_summary_

        Args:
            model_part (_type_): _description_
            layer_name (_type_): _description_
            key (_type_): _description_
            value (_type_): _description_

        Raises:
            ValueError: _description_
        """
        # Access backbone, neck or head
        model_part = self.config[model_part]
        
        # Go through all layers and check if the layer_name exists
        for index, layer in enumerate(model_part):
            
            # Accessing the layer name
            layer_name = layer[3]

            if layer_name == layer_name:
                if isinstance(layer[-1], dict) and key in layer[-1]:
                    layer[-1][key] = value
                    break
                else:
                    raise ValueError(f"Key {key} not found in layer {layer_name}.")
            else:
                raise ValueError(f"Layer {layer_name} not found in model configuration.")
            
    def set_training_param(self, key, value):
        """_summary_

        Args:
            key (_type_): _description_
            value (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method is not implemented yet.")
    
    def set_dataset_param(self, key, value):
        """_summary_

        Args:
            key (_type_): _description_
            value (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("This method is not implemented yet.")