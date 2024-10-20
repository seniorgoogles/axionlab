import os
import json
import shutil
import torch
import numpy as np
from includes_ml2 import *
from tqdm import tqdm
import time 

class ModelEvaluators:
    @staticmethod
    def evaluate_max_acc_drop_by_key_layerwise(model_config, weights_path, layer_str_list, key, max_acc_drop, save_dir, config_path=None):
        
        """
        # Initialize dataset, validator and model
        dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
        validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
        
        if config_path is not None:
            model_config = ConfigurationManager(config_path).config
        
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
        
        # Get baseline accuracy
        baseline_acc, _ = validator.validate(model)
        
        # Get baseline sparsity
        baseline_sparsity = get_sparsity_overview(model, layer_str_list)

        # Go through all layers
        for layer_str in layer_str_list:
            reload_index = 0
            
            for index, val in enumerate(value_range):
                
                # Set new value and rebuild model
                model_config = set_config_value(model_config, layer_str, key, val)
                model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
                
                # Validate model
                acc, _ = validator.validate(model)
                
                print(f"{baseline_acc=} {acc=} {max_acc_drop=}")
                
                # If accuracy drop is too high, reload previous model configuration
                if baseline_acc - acc > max_acc_drop:
                    reload_index = index - 1
                    print(f"Reload index: {reload_index}")
                    break
                else:
                    reload_index = index

            logger.debug(f"Layer: {layer_str} - {key} - {value_range[reload_index]}")
            # Set back to previous "good" model configuration 
            model_config = set_config_value(model_config, layer_str, key, value_range[reload_index])   
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
            acc, _ = validator.validate(model) 
        
        
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
        acc, loss = validator.validate(model)         
        curr_model_sparsity = get_model_sparsity(model, layer_str_list)
        
        save_config_path = f"{save_dir}/config.yaml"
        
        logging.info(f"Saving configuration to {save_config_path}")
        
        manager = ConfigurationManager(model_config)
        manager.write(save_config_path)
        
        return acc, loss, curr_model_sparsity
        """

class Report:
    
    def __init__(self) -> None:
        self.result = dict()
        
    def add_result_by_metric(self, optimization_scheme, metric, value, configuration):
        if optimization_scheme not in self.result:
            self.result[optimization_scheme] = dict()
        if metric not in self.result[optimization_scheme]:
            self.result[optimization_scheme][metric] = dict()
        self.result[optimization_scheme][metric][value] = configuration
        
    def set_baseline_acc(self, baseline_acc):
        self.result["baseline_acc"] = baseline_acc
    
    def set_model_layers(self, model_layers, params_per_layer):
        
        self.result["layers"] = dict()
        for layer, params in zip(model_layers, params_per_layer):
            
            mem_usage_kb = (params * 32) / 8 / 1024
            
            self.result["layers"][layer] = {
                "params": params,
                "memory_usage_kb": str(mem_usage_kb)
            }
            
        self.result["processing_sequence"] = model_layers
          
    # Write the report to a file as json
    def write(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True)
            
class PipelineManager:
    
    def __init__(self, dataset, validator, model_config_path, weights_config_path, layers, results_path) -> None:
        self.dataset = dataset
        self.validator = validator
        self.model_config_path = model_config_path
        self.weights_config_path = weights_config_path
        self.layers = layers
        self.results_path = results_path   

class StopWatch:
    
    @staticmethod
    def format_time(elapsed):
        """Convert time from seconds to HH:MM:SS format."""
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    
    @staticmethod
    def measure_time(func):
        def wrapper(step_num, total_steps, *args, **kwargs):
            start = time.time()  # Record start time
            result = func(step_num, total_steps, *args, **kwargs)  # Pass all arguments, including step_num and total_steps
            elapsed = time.time() - start  # Calculate elapsed time
            formatted_time = StopWatch.format_time(elapsed)  # Format the elapsed time in HH:MM:SS
            return formatted_time, result
        return wrapper

class Pipeline: 
    
    def __init__(self, pipeline_configuration, logger=None, report_path=None) -> None:
        self.pipeline_configuration = pipeline_configuration
        self.report = None
        self.logger = logger
        self.report_path = report_path
        
        if self.report_path != None:
            self._create_report()
        
    def delete_previous_results(self, save_dir):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.report = dict()
        
    def run(self, steps, step_params_list):
        total_steps = len(steps)  # Get the total number of steps
        for step_num, (step, params) in enumerate(zip(steps, step_params_list), start=1):
            formatted_time, result = self._run_step(step_num, total_steps, step, params)
            
            if self.logger != None:
                self.logger.info(f"Step {step_num}/{total_steps} - {step} - Time: {formatted_time} - Result: {result}")
            else:
                print(f"Step {step_num}/{total_steps} - {step} - Time: {formatted_time} - Result: {result}")
                
    def _create_report(self):
        self.report = Report()
        
        
        model = ModelBuilder().build(ModelTypes.JSC, config=base_model_config, weights_path=float_model_weight_path)
        layers, params = sort_layers_by_param_num(model, self.pipeline_configuration.layers)
                
        acc, _ = self.pipeline_configuration.validator.validate(model)

        self.report.set_baseline_acc(acc)
        self.report.set_model_layers(layers, params)
        
        self.report.write(self.report_path)
            
    def validate(self, *args, **kwargs):
        print("Validating model...")
        
    def evaluate(self, *args, **kwargs):
        print("====================================")
        print("Evaluating model...")
        print("====================================")

        eval_type = kwargs["eval_type"]

        if eval_type == "quantization":
            print("Quantization evaluation...")
            weight_bitwidth_range = kwargs["weights_range"]
            allowed_acc_drop = kwargs["allowed_acc_drop"]
            layer_str_list = kwargs["layers"]
            save_dir = kwargs["save_dir"]
            use_basemodel = False
            models_config_list = []
            
            if not isinstance(allowed_acc_drop, list):
                allowed_acc_drop_list = [allowed_acc_drop]
            else:
                allowed_acc_drop_list = allowed_acc_drop

            # Check if basemodel is used
            if "use_basemodel" in kwargs:
                use_basemodel = kwargs["use_basemodel"]
                models_config_list = [self.pipeline_configuration.model_config_path]
            
            for model_config in models_config_list:
                # Reload index
                reload_index = 0
                
                baseline_acc = self.pipeline_configuration.validator.validate(ModelBuilder().build(ModelTypes.JSC, 
                                                                                                config=model_config, 
                                                                                                weights_path=self.pipeline_configuration.weights_config_path))[0]

                for allowed_acc_drop in allowed_acc_drop_list:
                    # Enable quantization
                    base_model_config = set_config_value(self.pipeline_configuration.model_config_path, "dense", "weight_disable_quant", False)
                    eval_model_config = set_config_value(self.pipeline_configuration.model_config_path, "dense", "weight_disable_quant", False)
                    
                    for layer in layer_str_list:
                        
                        for index, weight in enumerate(weight_bitwidth_range):
                            print(f"Layer: {layer} - Weight: {weight}")
                            
                            eval_model_config = set_config_value(base_model_config, layer, "weight_bit_width", weight)
                            
                            model = ModelBuilder().build(ModelTypes.JSC, config=eval_model_config, weights_path=self.pipeline_configuration.weights_config_path)
                            acc =  self.pipeline_configuration.validator.validate(model)[0]

                        
                            # If accuracy drop is too high, reload previous model configuration
                            if baseline_acc - acc > allowed_acc_drop:
                                
                                if index > 0: 
                                    reload_index = index - 1
                                    
                                print(f"Reload index: {reload_index} - Acc: {acc}")
                                break
                            else:
                                reload_index = index
                        
                        print(f"{reload_index=}")
                        eval_model_config = set_config_value(base_model_config, layer, "weight_bit_width", weight_bitwidth_range[reload_index])
                    
                    # Create dir for saving data
                    os.makedirs(f"{save_dir}/allowed_acc_{allowed_acc_drop}", exist_ok=True)
                    
                    manager = ConfigurationManager(eval_model_config)
                    manager.write(f"{save_dir}/allowed_acc_{allowed_acc_drop}/config.yaml")
        ############################################################################################################
        # Sparse evaluation
        ############################################################################################################
        elif eval_type == "sparsity":
            print("Quantization evaluation...")
            sparsity_range = kwargs["sparsity_range"]
            allowed_acc_drop = kwargs["allowed_acc_drop"]
            layer_str_list = kwargs["layers"]
            save_dir = kwargs["save_dir"]
            use_basemodel = False
            models_config_list = []
            
            if not isinstance(allowed_acc_drop, list):
                allowed_acc_drop_list = [allowed_acc_drop]
            else:
                allowed_acc_drop_list = allowed_acc_drop

            # Check if basemodel is used
            if "use_basemodel" in kwargs:
                use_basemodel = kwargs["use_basemodel"]
                models_config_list = [self.pipeline_configuration.model_config_path]
            else:
                models_dir = kwargs["models_dir"]
                models_config_list = os.listdir(models_dir)
                print(models_config_list)
            
            for model_config in models_config_list:
                # Reload index
                reload_index = 0
                
                baseline_acc = self.pipeline_configuration.validator.validate(ModelBuilder().build(ModelTypes.JSC, 
                                                                                                config=model_config, 
                                                                                                weights_path=self.pipeline_configuration.weights_config_path))[0]

                for allowed_acc_drop in allowed_acc_drop_list:
                    
                    # Enable quantization
                    base_model_config = set_config_value(self.pipeline_configuration.model_config_path, "dense", "weight_disable_sparse", False)
                    eval_model_config = set_config_value(self.pipeline_configuration.model_config_path, "dense", "weight_disable_sparse", False)
                    
                    for layer in layer_str_list:
                        
                        for index, sparse_eps in enumerate(sparsity_range):
                            print(f"Layer: {layer} - Weight: {sparse_eps}")
                            
                            eval_model_config = set_config_value(base_model_config, layer, "weight_sparse_eps", sparse_eps)
                            
                            model = ModelBuilder().build(ModelTypes.JSC, config=eval_model_config, weights_path=self.pipeline_configuration.weights_config_path)
                            acc =  self.pipeline_configuration.validator.validate(model)[0]

                        
                            # If accuracy drop is too high, reload previous model configuration
                            if baseline_acc - acc > allowed_acc_drop:
                                
                                if index > 0: 
                                    reload_index = index - 1
                                    
                                print(f"Reload index: {reload_index} - Acc: {acc}")
                                break
                            else:
                                reload_index = index
                        
                        print(f"{reload_index=}")
                        eval_model_config = set_config_value(base_model_config, layer, "weight_sparse_eps", sparsity_range[reload_index])
                    
                    # Create dir for saving data
                    os.makedirs(f"{save_dir}/allowed_acc_{allowed_acc_drop}", exist_ok=True)
                    
                    manager = ConfigurationManager(eval_model_config)
                    manager.write(f"{save_dir}/allowed_acc_{allowed_acc_drop}/config.yaml")
        else:
            raise ValueError("Invalid evaluation type")
                        
    def retrain(self, *args, **kwargs):
        print("Retraining model...")        
        lr = kwargs["lr"]
        epochs = kwargs["epochs"]
        models_dir = kwargs["models_dir"]
        
        # List models folder 
        model_dir_list = os.listdir(models_dir)
        model_dir_list = [os.path.join(models_dir, model_dir) for model_dir in model_dir_list]
        
        for model_dir in model_dir_list:
            model_config = ConfigurationManager(os.path.join(model_dir, "config.yaml"))
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config.config, weights_path=self.pipeline_configuration.weights_config_path)
            
            optimizer = torch.optim.Adam(model.parameters(), lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
            
            trainer = Trainer(model, self.pipeline_configuration.dataset)
            trainer.train(model, None, self.pipeline_configuration.dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler, model_dir)
            
        
    def train(self, *args, **kwargs):
        print("Training model...")
    
    @StopWatch.measure_time
    def _run_step(self, step_num, total_steps, step, params):
        
        print(f"{params=}")
        
        switch = {
            "train": self.train,
            "validate": self.validate,
            "evaluate": self.evaluate,
            "retrain": self.retrain
        }
        # Call the function or return an invalid message
        return switch.get(step, lambda **kwargs: "Invalid option")(**params)

if __name__ == "__main__":
    
    base_dir = "tmp_data/experiment_quant_first"
        
    # Remove base directory if it exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)    
        
    base_model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl_updated_quant.yaml"
    float_model_weight_path = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_331/best_weights.pth"
    
    processing_sequence = ["dense2", "dense3", "dense4", "dense1", "dense5"]
    
    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=base_model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
        
    config_manager = PipelineManager(dataset=dataset, 
                                   validator=validator, 
                                   model_config_path=base_model_config, 
                                   weights_config_path=float_model_weight_path, 
                                   layers=processing_sequence,
                                   results_path="report.json")
    
    file_logger = logging.getLogger("file_logger")
    file_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("pipeline.log")
    file_handler.setLevel(logging.DEBUG)
    file_logger.addHandler(file_handler)
    
    file_logger=None
    
    pipeline = Pipeline(config_manager, logger=file_logger, report_path="results.json")

    # Define the steps and their corresponding parameters
    steps = [
        "evaluate", 
        "retrain",
        "evaluate",
        "retrain"
    ]
    step_params_list = [
        {   
            "step_name": "0_evaluate_quantization",
            "eval_type": "quantization",     
            "use_basemodel": True,
            "weights_range": [i for i in range(16, 15, -1)],
            "allowed_acc_drop": [0.25, 0.5, 1.0],
            "layers": processing_sequence,
            "save_dir": f"{base_dir}/{0}_quantization",
        }, 
        {    
            "step_name": "1_retrain_after_quantization",
            "lr": 0.00009349,
            "epochs": 1,
            "models_dir": f"{base_dir}/{0}_quantization",
        }, 
        {   
            "step_name": "2_evaluate_sparsity",
            "use_basemodel": False,
            "sparsity_range": np.arange(0.0, 1.0, 0.05),
            "eval_type": "sparsity",
            "allowed_acc_drop": [0.25, 0.5, 1.0],
            "layers": processing_sequence,
            "models_dir": f"{base_dir}/{0}_quantization",
            "save_dir": f"{base_dir}/{1}_sparsity",
        }, 
        {    
            "step_name": "1_retrain_after_quantization",
            "lr": 0.00009349,
            "epochs": 1,
            "models_dir": f"{base_dir}/{1}_sparsity"
        }, 
    ]  # You can add actual parameters as needed

    # Run the pipeline
    pipeline.run(steps, step_params_list)
    



"""
def retrain(config_dir, weights_path, lr, epochs): 
    
    config = os.path.join(config_dir, "config.yaml")
    # Initialize the configuration manager
    config_manager = ConfigurationManager(config)
    
    # Load the model
    model = ModelBuilder().build(ModelTypes.JSC, config=config_manager.config, weights_path=weights_path)
    
    # Load the dataset and validate base model
    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=config_manager.config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    # Validate
    start_acc, loss = validator.validate(model)

    # Initialize the trainer and run training
    trainer = Trainer(model, dataset)
    
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
    
    name = config.replace('.yaml', '').split('/')[-1]
    save_dir = f"{config_dir}"
    
    trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler, save_dir)
    
    # Reload model with best weights
    model = ModelBuilder().build(ModelTypes.JSC, config=config_manager.config, weights_path=f"{save_dir}/best_weights.pth")

    # Validate the model
    acc, loss = validator.validate(model)

    # Return accuracy and config name
    return acc, loss

def evaluate_max_acc_drop_by_key_layerwise(model_config, weights_path, layer_str_list, key, value_range, max_acc_drop, logger, save_dir, config_path=None):
    
    # Initialize dataset, validator and model
    dataset = DatasetBuilder.build(DatasetTypes.JSC, config=model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    
    if config_path is not None:
        model_config = ConfigurationManager(config_path).config
    
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
    
    # Get baseline accuracy
    baseline_acc, _ = validator.validate(model)
    
    # Get baseline sparsity
    baseline_sparsity = get_sparsity_overview(model, layer_str_list)

    # Go through all layers
    for layer_str in layer_str_list:
        reload_index = 0
        
        for index, val in enumerate(value_range):
            
            # Set new value and rebuild model
            model_config = set_config_value(model_config, layer_str, key, val)
            model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
            
            # Validate model
            acc, _ = validator.validate(model)
            
            print(f"{baseline_acc=} {acc=} {max_acc_drop=}")
            
            # If accuracy drop is too high, reload previous model configuration
            if baseline_acc - acc > max_acc_drop:
                reload_index = index - 1
                print(f"Reload index: {reload_index}")
                break
            else:
                reload_index = index

        logger.debug(f"Layer: {layer_str} - {key} - {value_range[reload_index]}")
        # Set back to previous "good" model configuration 
        model_config = set_config_value(model_config, layer_str, key, value_range[reload_index])   
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
        acc, _ = validator.validate(model) 
      
      
    model = ModelBuilder().build(ModelTypes.JSC, config=model_config, weights_path=weights_path)
    acc, loss = validator.validate(model)         
    curr_model_sparsity = get_model_sparsity(model, layer_str_list)
    
    save_config_path = f"{save_dir}/config.yaml"
    
    logging.info(f"Saving configuration to {save_config_path}")
    
    manager = ConfigurationManager(model_config)
    manager.write(save_config_path)
    
    return acc, loss, curr_model_sparsity

# Define a generic ConfigManager for handling paths and configs
class ConfigManager:
    def __init__(self, config_path, weight_path):
        self.config_path = config_path
        self.weight_path = weight_path

    def load_model_and_dataset(self):
        model = ModelBuilder().build(ModelTypes.JSC, config=self.config_path, weights_path=self.weight_path)
        dataset = DatasetBuilder.build(DatasetTypes.JSC, config=self.config_path)
        return model, dataset

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        # Save relevant configuration information here
        # For now, let's say we save the config path for reference
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump({"config_path": self.config_path}, f, indent=4)

# Define a class to manage the entire pipeline
class Pipeline:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.report = None   
        
        self.baseline_acc = 0.0
        
    def delete_previous_results(self, save_dir):
        print(f"Deleting previous results {save_dir}...")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.report = dict()
        
    def validate_baseline(self):
        model = ModelBuilder().build(ModelTypes.JSC, config=self.config_manager.config_path, weights_path=self.config_manager.weight_path)
        dataset = DatasetBuilder.build(DatasetTypes.JSC, config=self.config_manager.config_path)
        validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
        acc, loss = validator.validate(model)
            
        return acc, loss
            
    def evaluate(self, eval_type ,model_config, model_weights, layer_str_list, save_dir, sparsity_range=None, weights_range=None, allowed_acc_drop=0.0):
        ###Generalized evaluation for both sparsity and quantization
        logger.debug(f"Evaluating {eval_type}")
        
        if eval_type == "sparsity":
            model_config = set_config_value(model_config, "dense", "weight_disable_sparse", False)
            acc, loss, current_model_sparsity = evaluate_max_acc_drop_by_key_layerwise(model_config, model_weights, layer_str_list,  "weight_sparse_eps", sparsity_range, allowed_acc_drop, logger, save_dir)
            logger.info(f"Sparsity evaluation: {acc=} {allowed_acc_drop=} {current_model_sparsity=}")
        elif eval_type == "quantization":
            model_config = set_config_value(base_model_config, "dense", "weight_disable_quant", False)
            acc, loss, current_model_sparsity = evaluate_max_acc_drop_by_key_layerwise(model_config, model_weights, layer_str_list,  "weight_bit_width", weights_range, allowed_acc_drop, logger, save_dir)
            logger.info(f"Quantization evaluation: {acc}= {allowed_acc_drop=} {current_model_sparsity=}")

        return acc, loss, current_model_sparsity
    
    def retrain(self, lr, epochs, config_dir, model_weights):
        acc, loss = retrain(config_dir, model_weights, lr, epochs)
        return acc, loss
    
    def execute_step(self, step, params, logger=None):

        if step == "validate_baseline":
            acc, loss = self.validate_baseline(**params)
            logger.info(f"Validation baseline accuracy: {acc}% loss: {loss}")
            
        elif step == "evaluate":
            acc, loss, model_sparsity = self.evaluate(**params)
            logger.info(f"Validation baseline accuracy: {acc}% loss: {loss}")
            
        elif step == "retrain":
            acc, loss = self.retrain(**params)
            logger.info(f"Validation baseline accuracy: {acc}% loss: {loss}")
        
# Main orchestrator for running the pipeline
if __name__ == "__main__":
    # Configuration
    base_model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl_updated_quant.yaml"
    float_model_weight_path = f"/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_331/best_weights.pth"
    
    base_dir = "tmp_data/experiment_quant_first"
    base_line_acc = 0.0
    
    # Remove base directory if it exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # Make sure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    lr = 0.00009349
    epochs = 30

    # Define ranges for evaluation
    sparsity_range = np.arange(0.0, 1.0, 0.05)
    word_width_range = range(16, 3, -1)

    max_allowed_acc_drop = 0.5
    acc_drop_range_raw = np.arange(0.0, max_allowed_acc_drop + 0.25, 0.25)
    allowed_acc_drop_range = [round(x, 2) for x in acc_drop_range_raw]
    
    # Create a ConfigManager instance
    config_manager = ConfigManager(base_model_config, float_model_weight_path)
    
    layer_str_list = ["dense1", "dense2", "dense3", "dense4", "dense5"]

    # Create the pipeline
    pipeline = Pipeline(config_manager)

    # Define the order of steps in the process
    steps_order = [
        "validate_baseline",
        "evaluate",
        "retrain",
        "evaluate",
        "retrain",
    ]
    
    steps_params = [
        {},
        {
            "model_config": base_model_config,
            "model_weights": float_model_weight_path,
            "weights_range": word_width_range,
            "eval_type": "quantization",
            "layer_str_list": layer_str_list,
            "save_dir": f"{base_dir}/{0}_quantization",
        },
        {
            "lr": lr, 
            "epochs": 1, 
            "config_dir": f"{base_dir}/{0}_quantization",
            "model_weights": float_model_weight_path,
        },
        {
            "model_config": base_model_config,
            "model_weights": float_model_weight_path,
            "weights_range": word_width_range,
            "eval_type": "sparsity",
            "layer_str_list": layer_str_list,
            "config_dir": f"{base_dir}/{0}_quantization",
            "save_dir": f"{base_dir}/{1}_sparsity",
        },
        {
            "lr": lr, 
            "epochs": 1, 
            "config_dir": f"{base_dir}/{1}_sparsity",
            "model_weights": float_model_weight_path,
        },
    ]

    # Execute all steps in order
    save_dir_base = "results"
    
    # Add processing bar here  
    
    overall_steps = (len(steps_order) - 1) * len(allowed_acc_drop_range) if "validate_baseline" in steps_order else len(steps_order) * len(acc_drop_range)
    current_step = 1
    
    # Setup logging, which writes into a file in the base directory
    logging.basicConfig(filename=f"{base_dir}/pipeline.log", level=logging.DEBUG)

    # Make logging print to console as well
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    
    logger = logging.getLogger()
    
    # Format logs to include time and with  time % [LOGGING_LEVEL] % message
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    
    prev_base_dir = None

    for i, (step, params) in enumerate(zip(steps_order, steps_params)):
        current_step += 1
        logger.info(f"Step {i+1}/{len(steps_order)} - {step}")

        # Single steps
        if step == "validate_baseline":
            pipeline.execute_step(step, params, logger)
            
        elif step == "delete_previous_results":
            pipeline.delete_previous_results(f"{save_dir_base}")

        else: 
            
            # Create a copy of params
            pipeline_params = params.copy()
            base_save_dir = ""
            # Loop over all accuracy drop values
            for allowed_acc_drop in allowed_acc_drop_range:
                logger.info(f"Step {i+1}/{len(steps_order)} - {step} - {current_step}/{overall_steps}")
                
                if "save_dir" in params:
                    save_dir = os.path.join(params["save_dir"], f"allowed_acc_drop_{allowed_acc_drop}")
                    pipeline_params["save_dir"] = save_dir
                    # Add allowed_acc_drop to the params
                    pipeline_params["allowed_acc_drop"] = allowed_acc_drop
                if "config_dir" in params:
                    config_dir = os.path.join(params["config_dir"], f"allowed_acc_drop_{allowed_acc_drop}")
                    pipeline_params["config_dir"] = config_dir
                
                # Create save_dir if it does not exist
                os.makedirs(save_dir, exist_ok=True)
                
                pipeline.execute_step(step, pipeline_params, logger)
"""
