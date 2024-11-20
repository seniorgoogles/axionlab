import os
import json
import shutil
import torch
import numpy as np
from includes_ml2 import *
from tqdm import tqdm
import time 

class Report:
    
    def __init__(self) -> None:
        self.result = dict()
        
    def add_result_by_metric(self, pipeline_step, exp_name, model_info):
        
        # If key pipleline_step does not exist, create it
        if pipeline_step not in self.result:
            self.result[pipeline_step] = dict()
            self.result[pipeline_step]["experiments"] = dict()
            
        # Append the result to the experiments list
        self.result[pipeline_step]["experiments"][exp_name] = model_info

    def set_baseline_acc(self, baseline_acc):
        self.result["baseline_acc"] = baseline_acc
    
    def set_model_layers(self, model_layers, params_per_layer):
        
        self.result["layers"] = dict()
        for layer, params in zip(model_layers, params_per_layer):
            
            mem_usage_kb = (params * 32) / 8 / 1024
            
            self.result["layers"][layer] = {
                "params": params,
                "memory_usage_kb": mem_usage_kb
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

        step_name = kwargs["step_name"]
        eval_key = kwargs["eval_key"]
        eval_range = kwargs["eval_range"]
        weight_prune_first = kwargs["weight_prune_first"]
        use_basemodel = kwargs["use_basemodel"]
        allowed_acc_drop = kwargs["allowed_acc_drop"]
        layers = kwargs["layers"]
        models_dir = kwargs["models_dir"]
        save_dir = kwargs["save_dir"]
                
        # Convert allowed_acc_drop to list if it is not
        if not isinstance(allowed_acc_drop, list):
            allowed_acc_drop_list = [allowed_acc_drop]
        else:
            allowed_acc_drop_list = allowed_acc_drop
            
        # Check if basemodel is used, if not use models from models_dir
        if use_basemodel:
            models_config_list = [self.pipeline_configuration.model_config_path]
            model_weights_list = [self.pipeline_configuration.weights_config_path]
        else:
            models_config_list = [os.path.join(models_dir, model_dir, "config.yaml") for model_dir in os.listdir(models_dir)]
            model_weights_list = [os.path.join(models_dir, model_dir, "best_weights.pth") for model_dir in os.listdir(models_dir)]
        
        exp_index = 0
        
        # Iiterate over all models
        for model_config, model_weights in zip(models_config_list, model_weights_list):
            # Reload index
            reload_index = 0
            
            # Get baseline accuracy
            baseline_acc = self.pipeline_configuration.validator.validate(ModelBuilder().build(ModelTypes.JSC, 
                                                                                            config=model_config, 
                                                                                            weights_path=model_weights))[0]
        
            # Iterate over all allowed_acc_drop values
            for allowed_acc_drop in allowed_acc_drop_list:
                
                # Load base model configuration
                eval_model_config = ConfigurationManager(model_config).config          
                bitwidth_weights_list = []
                
                # Iterate over all layers
                for layer in layers:
                    # Iterate over all values in the range
                    
                    # Setup model configuration
                    eval_model_config = set_config_value(eval_model_config, layer, "weight_prune_first", weight_prune_first)
                    
                    # If weight_sparse_eps is set, enable pruning, else if weight_bit_width is set, enable quantization
                    if eval_key == "weight_sparse_eps":
                        eval_model_config = set_config_value(eval_model_config, layer, "weight_disable_sparse", False)
                    elif eval_key == "weight_bit_width":
                        eval_model_config = set_config_value(eval_model_config, layer, "weight_disable_quant", False)
                    else:
                        raise ValueError("Invalid eval_key")

                    # Evaluation loop                  
                    for index, val in enumerate(eval_range):
                        
                    
                        print(f"Layer: {layer} - {eval_key}: {val}")

                        eval_model_config = set_config_value(eval_model_config, layer, eval_key, val)
                        
                        model = ModelBuilder().build(ModelTypes.JSC, config=eval_model_config, weights_path=model_weights)
                        acc =  self.pipeline_configuration.validator.validate(model)[0]
                        
                        # If accuracy drop is too high, reload previous model configuration
                        if baseline_acc - acc > allowed_acc_drop:
                            if index > 0: 
                                reload_index = index - 1

                            break
                        else:
                            reload_index = index
                    
                    # Set back to previous "good" model configuration 
                    eval_model_config = set_config_value(eval_model_config, layer, eval_key, eval_range[reload_index])   
                    bitwidth_weights_list.append(get_config_value(eval_model_config, layer, "weight_bit_width"))
                    acc =  self.pipeline_configuration.validator.validate(model)[0]
                                        

                layer_params_list = [get_weights_per_layer(model, layer) for layer in layers]
                layer_params_zero_value_list = [get_weights_zero_value_per_layer(model, layer) for layer in layers]
                layer_usagage_kb_max_list = [(params * 32) / 8 / 1024 for params in layer_params_list]
                layer_usage_kb_curr_list = [(params * bitwidth) / 8 / 1024 for params, bitwidth in zip(layer_params_list, bitwidth_weights_list)]
                
                dict_layers = dict()
                for layer, params, zero_params, max_usage, curr_usage, bitwidth_weights in zip(layers, layer_params_list, layer_params_zero_value_list, layer_usagage_kb_max_list, layer_usage_kb_curr_list, bitwidth_weights_list):
                    dict_layers[layer] = {
                        "params": params,
                        "zero_params": zero_params,
                        "max_usage_kb": max_usage,
                        "curr_usage_kb": curr_usage,
                        "bitwidth": bitwidth_weights
                    }
                    
                # Write report to file
                self.report.add_result_by_metric(step_name, f"exp_{exp_index}", {
                    "allowed_accuracy_drop": allowed_acc_drop,
                    "accuracy": acc,
                    "layers": dict_layers
                })
                # Create dir for saving data
                os.makedirs(f"{save_dir}/allowed_acc_{allowed_acc_drop}", exist_ok=True)
                self.report.write(self.report_path)
                
                manager = ConfigurationManager(eval_model_config)
                manager.write(f"{save_dir}/allowed_acc_{allowed_acc_drop}/config.yaml")

                exp_index += 1
                        
    def retrain(self, *args, **kwargs):
        
        print("Retraining model...")        
        lr = kwargs["lr"]
        epochs = kwargs["epochs"]
        
        config_path = kwargs["config_path"]
        weight_path = kwargs["weight_path"]
        save_dir = kwargs["save_dir"]

            
        # Check if save_dir exists, if it exists, delete it and recreate it
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            
        os.makedirs(save_dir, exist_ok=True)

        step_name = kwargs["step_name"]

        model_config = ConfigurationManager(config_path)
        model = ModelBuilder().build(ModelTypes.JSC, config=model_config.config, weights_path=weight_path)
        
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True)
        
        trainer = Trainer(model, self.pipeline_configuration.dataset)
        trainer.train(model, None, self.pipeline_configuration.dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler, save_dir)
        
        print(f"{lr=}")
        trainer.train(model, None, self.pipeline_configuration.dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs - 10, 200, scheduler, save_dir)
        
        lr = lr + (lr * 0.005)
        print(f"{lr=}")
        trainer.train(model, None, self.pipeline_configuration.dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs - 10, 200, scheduler, save_dir)
        
        lr = lr + (lr * 0.005)
        print(f"{lr=}")
        trainer.train(model, None, self.pipeline_configuration.dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs - 20, 200, scheduler, save_dir)
        
        lr = lr +(lr * 0.005)        
        print(f"{lr=}")
        trainer.train(model, None, self.pipeline_configuration.dataset, torch.nn.CrossEntropyLoss(), optimizer, lr, epochs, 200, scheduler, save_dir)

        
        acc, _ = self.pipeline_configuration.validator.validate(model)
        
        # Write report to file
        self.report.add_result_by_metric(step_name, f"exp", {
            "accuracy": acc,
        })
        
        self.report.write(self.report_path)

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
        
    # Remove base directory if it exists, if user input is y else not
    if os.path.exists(base_dir):
        user_input = input("Directory exists, do you want to remove it? (y/n)")
        if user_input == "y":
            shutil.rmtree(base_dir)

    base_model_config = f"{parent_directory}/configs/jsc/quant_jsc_xl_weight_bias_quant.yaml"
    float_model_weight_path = "/home/fry/new_repo/synapselab/train/jsc_xl/run_4/best_weights.pth"
    
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
    #    "evaluate", 
    #    "retrain",
    #    "evaluate",
        "retrain"
    ]
    
    '''
        step_name = kwargs["step_name"]
        eval_key = kwargs["eval_key"]
        eval_range = kwargs["eval_range"]
        use_basemodel = kwargs["use_basemodel"]
        allowed_acc_drop = kwargs["allowed_acc_drop"]
        layers = kwargs["layers"]
        models_dir = kwargs["models_dir"]
        save_dir = kwargs["save_dir"]
    '''
    step_params_list = [
        {   
            "step_name": "0_evaluate_quantization",
            "eval_key": "weight_bit_width",
            "eval_range": [i for i in range(16, 2, -1)],
            "weight_prune_first": False,
            "use_basemodel": True,
            "allowed_acc_drop": [60.0], #np.arange(0.0, 1.0, 1.0).tolist(),
            "layers": processing_sequence,
            "models_dir": None,
            "save_dir": f"{base_dir}/{0}_quantization",
        }, 
        {    
            "step_name": "1_retrain_after_quantization",
            "lr": 0.00019349,
            "epochs": 40,
            "models_dir": f"{base_dir}/{0}_quantization",
        }, 
        {   
            "step_name": "2_evaluate_pruning",
            "eval_key": "weight_sparse_eps",
            "eval_range": np.arange(0.0, 1.0, 0.01),
            "weight_prune_first": False,
            "use_basemodel": False,
            "allowed_acc_drop": [3.0], #np.arange(0.0, 1.0, 1.0).tolist(),
            "layers": processing_sequence,
            "models_dir": f"{base_dir}/{0}_quantization",
            "save_dir": f"{base_dir}/{1}_pruning",
        }, 
        {    
            "step_name": "3_retrain_after_pruning",
            "lr": 0.00003349,
            "epochs": 100,
            "models_dir": f"{base_dir}/{1}_pruning"
        }, 
    ]  # You can add actual parameters as needed

    step_params_list = [
        {    
            "step_name": "0_init_train",
            "lr": 0.02063015025,
            "epochs": 35,
            "config_path": f"/home/fry/new_repo/synapselab/weights/jsc/config.yaml",
            "weight_path": f"/home/fry/new_repo/synapselab/weights/jsc/best_weights.pth",
            "save_dir": f"{base_dir}/0_init_train",
        },    
    ]
    # Run the pipeline
    pipeline.run(steps, step_params_list)

    #model = ModelBuilder().build(ModelTypes.JSC, config="/home/fry/new_repo/synapselab/tmp_data/experiment_quant_first/1_pruning/allowed_acc_3.0/config.yaml", weights_path="/home/fry/new_repo/synapselab/tmp_data/experiment_quant_first/1_pruning/allowed_acc_3.0/best_weights.pth")
    #validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())
    #acc, _ = validator.validate(model)
    
    