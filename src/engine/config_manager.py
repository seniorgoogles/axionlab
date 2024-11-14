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
                
    def generate_pytorch_model(self, name=None, output_path=None):
        
        
        if name is None:
            # Extract the model name from the config, using it as the class name
            model_name = self.config.get("name", "GeneratedModel")  # Default to "GeneratedModel" if "name" is not provided
        else:
            model_name = name
            
        # Begin defining the model string with dynamic class name
        model_str = "import torch\n"
        model_str += "import torch.nn as nn\n\n\n"
        model_str += f"class {model_name}(nn.Module):\n"
        model_str += f"\tdef __init__(self):\n"
        model_str += f"\t\tsuper({model_name}, self).__init__()\n"

        # Function to parse and add layers from the configuration to the model string
        def add_layers(section):
            layer_defs = ""
            for layer_def in section:
                _, _, layer_type, layer_name, layer_args = layer_def
                layer_args_str = ", ".join(f"{k}={repr(v)}" for k, v in layer_args.items())
                layer_defs += f"\t\tself.{layer_name} = nn.{layer_type}({layer_args_str})\n"
                
            layer_defs += "\n"
            
            return layer_defs

        # Add backbone, neck, and head layers if they exist in the config
        model_str += add_layers(self.config["backbone"])
        if "neck" in self.config:
            model_str += add_layers(self.config["neck"])
        if "head" in self.config:
            model_str += add_layers(self.config["head"])

        # Define the forward function with each layer called sequentially
        model_str += "\tdef forward(self, x):\n"

        # Add each layer to the forward pass
        for layer_def in self.config.get("backbone", []):
            _, _, _, layer_name, _ = layer_def
            model_str += f"\t\tx = self.{layer_name}(x)\n"
        if "neck" in self.config:
            for layer_def in self.config["neck"]:
                _, _, _, layer_name, _ = layer_def
                model_str += f"\t\tx = self.{layer_name}(x)\n"
        if "head" in self.config:
            for layer_def in self.config["head"]:
                _, _, _, layer_name, _ = layer_def
                model_str += f"\t\tx = self.{layer_name}(x)\n"

        # End of forward method
        model_str += "\n\t\treturn x\n\n"
        
        # Add load weights function
        model_str += "\tdef load_weights(self, weights_path):\n"
        model_str += "\t\tself.load_state_dict(torch.load(weights_path))\n"

        # Write the generated model to a Python file with the model name
        if output_path:
            filename = output_path
        else:
            filename = f"{model_name.lower()}.py"
            
        with open(filename, "w") as f:
            f.write(model_str)

        print(f"Model written to {filename}")
    
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