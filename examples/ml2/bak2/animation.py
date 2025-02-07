import os
import json
import shutil
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from includes_ml2 import (
    ModelBuilder,
    ModelTypes,
    Validator,
    DatasetBuilder,
    DatasetTypes,
    ConfigurationManager,
    Trainer,
    set_config_value,
    get_config_value,
    sort_layers_by_param_num,
    get_weights_per_layer,
    get_weights_zero_value_per_layer,
    parent_directory,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Report:
    """Class to handle reporting of pipeline results."""

    def __init__(self) -> None:
        """Initialize the Report with an empty result dictionary."""
        self.result: Dict[str, Any] = {}

    def add_result_by_metric(
        self, pipeline_step: str, exp_name: str, model_info: Dict[str, Any]
    ) -> None:
        """
        Add results to the report grouped by metric.

        Args:
            pipeline_step (str): The name of the pipeline step.
            exp_name (str): The name of the experiment.
            model_info (dict): Information about the model.
        """
        step = self.result.setdefault(pipeline_step, {"experiments": {}})
        step["experiments"][exp_name] = model_info

    def set_baseline_acc(self, baseline_acc: float) -> None:
        """
        Set the baseline accuracy in the report.

        Args:
            baseline_acc (float): The baseline accuracy.
        """
        self.result["baseline_acc"] = baseline_acc

    def set_model_layers(
        self, model_layers: List[str], params_per_layer: List[int]
    ) -> None:
        """
        Set the model layers and their parameters in the report.

        Args:
            model_layers (list): List of layer names.
            params_per_layer (list): List of parameters per layer.
        """
        self.result["layers"] = {}
        for layer, params in zip(model_layers, params_per_layer):
            mem_usage_kb = (params * 32) / 8 / 1024
            self.result["layers"][layer] = {
                "params": params,
                "memory_usage_kb": mem_usage_kb,
            }
        self.result["processing_sequence"] = model_layers

    def write(self, file_path: str) -> None:
        """
        Write the report to a file as JSON.

        Args:
            file_path (str): The path to the file where the report will be saved.
        """
        with open(file_path, "w") as f:
            json.dump(self.result, f, indent=4, sort_keys=True)


class PipelineManager:
    """Class to manage pipeline configurations."""

    def __init__(
        self,
        dataset: Any,
        validator: Validator,
        model_config_path: str,
        weights_config_path: str,
        layers: List[str],
        results_path: str,
    ) -> None:
        """
        Initialize the PipelineManager.

        Args:
            dataset: The dataset object.
            validator: The validator object.
            model_config_path (str): Path to the model configuration file.
            weights_config_path (str): Path to the model weights file.
            layers (list): List of layer names.
            results_path (str): Path to the results file.
        """
        self.dataset = dataset
        self.validator = validator
        self.model_config_path = model_config_path
        self.weights_config_path = weights_config_path
        self.layers = layers
        self.results_path = results_path


class StopWatch:
    """Utility class for measuring execution time."""

    @staticmethod
    def format_time(elapsed: float) -> str:
        """
        Convert time from seconds to HH:MM:SS format.

        Args:
            elapsed (float): Elapsed time in seconds.

        Returns:
            str: Formatted time as a string.
        """
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    @staticmethod
    def measure_time(func: Callable) -> Callable:
        """
        Decorator to measure the execution time of a function.

        Args:
            func (Callable): The function to measure.

        Returns:
            Callable: The wrapped function.
        """

        def wrapper(step_num: int, total_steps: int, *args, **kwargs):
            start = time.time()
            result = func(step_num, total_steps, *args, **kwargs)
            elapsed = time.time() - start
            formatted_time = StopWatch.format_time(elapsed)
            return formatted_time, result

        return wrapper


class Pipeline:
    """Class to manage the execution of pipeline steps."""

    def __init__(
        self,
        pipeline_configuration: PipelineManager,
        logger: Optional[logging.Logger] = None,
        report_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the Pipeline.

        Args:
            pipeline_configuration (PipelineManager): The pipeline configuration manager.
            logger (Optional[logging.Logger]): Logger for logging information.
            report_path (Optional[str]): Path to save the report.
        """
        self.pipeline_configuration = pipeline_configuration
        self.logger = logger
        self.report_path = report_path
        self.report = Report() if report_path else None

        if self.report_path:
            self._create_report()

    def delete_previous_results(self, save_dir: str) -> None:
        """
        Delete previous results from the specified directory.

        Args:
            save_dir (str): The directory to clean up.
        """
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.report = Report()

    def run(self, steps: List[Dict[str, Any]]) -> None:
        """
        Run the pipeline with the given steps.

        Args:
            steps (List[Dict[str, Any]]): A list of steps with their parameters.
        """
        total_steps = len(steps)
        for step_num, step_info in enumerate(steps, start=1):
            step_name = step_info["name"]
            params = step_info["params"]
            try:
                formatted_time, result = self._run_step(
                    step_num, total_steps, step_name, params
                )
                message = (
                    f"Step {step_num}/{total_steps} - {step_name} - "
                    f"Time: {formatted_time} - Result: {result}"
                )
                if self.logger:
                    self.logger.info(message)
                else:
                    print(message)
            except Exception as e:
                error_message = f"Error in step {step_name}: {e}"
                if self.logger:
                    self.logger.error(error_message)
                else:
                    print(error_message)

    def _create_report(self) -> None:
        """Create the initial report with baseline accuracy and model layers."""
        model = ModelBuilder().build(
            ModelTypes.JSC,
            config=self.pipeline_configuration.model_config_path,
            weights_path=self.pipeline_configuration.weights_config_path,
        )
        layers, params = sort_layers_by_param_num(
            model, self.pipeline_configuration.layers
        )

        baseline_acc, _ = self.pipeline_configuration.validator.validate(model)
        self.report.set_baseline_acc(baseline_acc)
        self.report.set_model_layers(layers, params)
        self.report.write(self.report_path)

    def validate(self, **kwargs) -> None:
        """Validate the model."""
        logger.info("Validating model...")

    def evaluate(self, **kwargs) -> None:
        """
        Evaluate the model based on the given parameters.

        Args:
            kwargs: Keyword arguments containing evaluation parameters.
        """
        logger.info("Evaluating model...")
        step_name = kwargs["step_name"]
        eval_key = kwargs["eval_key"]
        eval_range = kwargs["eval_range"]
        weight_prune_first = kwargs.get("weight_prune_first", False)
        use_basemodel = kwargs["use_basemodel"]
        allowed_acc_drop_list = (
            kwargs["allowed_acc_drop"]
            if isinstance(kwargs["allowed_acc_drop"], list)
            else [kwargs["allowed_acc_drop"]]
        )
        layers = kwargs["layers"]
        models_dir = kwargs["models_dir"]
        save_dir = kwargs["save_dir"]

        models_config_list, model_weights_list = self._get_model_list(
            use_basemodel, models_dir
        )

        exp_index = 0
        for model_config, model_weights in zip(models_config_list, model_weights_list):
            baseline_acc = self._get_baseline_accuracy(model_config, model_weights)
            for allowed_acc_drop in allowed_acc_drop_list:
                eval_model_config = ConfigurationManager(model_config).config
                bitwidth_weights_list = []
                for layer in layers:
                    eval_model_config = set_config_value(
                        eval_model_config, layer, "weight_prune_first", weight_prune_first
                    )
                    if eval_key == "weight_sparse_eps":
                        eval_model_config = set_config_value(
                            eval_model_config, layer, "weight_disable_sparse", False
                        )
                    elif eval_key == "weight_bit_width":
                        eval_model_config = set_config_value(
                            eval_model_config, layer, "weight_disable_quant", False
                        )
                    else:
                        raise ValueError("Invalid eval_key")

                    reload_index = 0
                    for index, val in enumerate(eval_range):
                        logger.info(f"Layer: {layer} - {eval_key}: {val}")
                        eval_model_config = set_config_value(
                            eval_model_config, layer, eval_key, val
                        )
                        model = ModelBuilder().build(
                            ModelTypes.JSC,
                            config=eval_model_config,
                            weights_path=model_weights,
                        )
                        acc = self.pipeline_configuration.validator.validate(model)[0]
                        if baseline_acc - acc > allowed_acc_drop:
                            if index > 0:
                                reload_index = index - 1
                            break
                        else:
                            reload_index = index

                    eval_model_config = set_config_value(
                        eval_model_config,
                        layer,
                        eval_key,
                        eval_range[reload_index],
                    )
                    bitwidth_weights_list.append(
                        get_config_value(eval_model_config, layer, "weight_bit_width")
                    )
                    acc = self.pipeline_configuration.validator.validate(model)[0]

                layer_info = self._gather_layer_info(
                    model, layers, bitwidth_weights_list
                )
                self.report.add_result_by_metric(
                    step_name,
                    f"exp_{exp_index}",
                    {
                        "allowed_accuracy_drop": allowed_acc_drop,
                        "accuracy": acc,
                        "layers": layer_info,
                    },
                )

                save_path = os.path.join(save_dir, f"allowed_acc_{allowed_acc_drop}")
                os.makedirs(save_path, exist_ok=True)
                self.report.write(self.report_path)

                manager = ConfigurationManager(eval_model_config)
                manager.write(os.path.join(save_path, "config.yaml"))

                exp_index += 1

    def retrain(self, **kwargs) -> None:
        """
        Retrain the model based on the given parameters.

        Args:
            kwargs: Keyword arguments containing retraining parameters.
        """
        logger.info("Retraining model...")
        lr = kwargs["lr"]
        epochs = kwargs["epochs"]
        models_dir = kwargs["models_dir"]
        step_name = kwargs["step_name"]

        model_dirs = [
            os.path.join(models_dir, d)
            for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]

        exp_index = 0
        for model_dir in model_dirs:
            model_config_path = os.path.join(model_dir, "config.yaml")
            model_config = ConfigurationManager(model_config_path)
            model = ModelBuilder().build(
                ModelTypes.JSC,
                config=model_config.config,
                weights_path=self.pipeline_configuration.weights_config_path,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=15, verbose=True
            )

            trainer = Trainer(model, self.pipeline_configuration.dataset)
            trainer.train(
                model=model,
                criterion=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer,
                lr=lr,
                epochs=epochs,
                eval_interval=200,
                scheduler=scheduler,
                save_dir=model_dir,
            )

            acc, _ = self.pipeline_configuration.validator.validate(model)
            self.report.add_result_by_metric(
                step_name, f"exp_{exp_index}", {"accuracy": acc}
            )
            self.report.write(self.report_path)

            exp_index += 1

    def train(self, **kwargs) -> None:
        """Train the model."""
        logger.info("Training model...")

    @StopWatch.measure_time
    def _run_step(
        self, step_num: int, total_steps: int, step_name: str, params: Dict[str, Any]
    ) -> Any:
        """
        Execute a single pipeline step.

        Args:
            step_num (int): The current step number.
            total_steps (int): The total number of steps.
            step_name (str): The name of the step.
            params (dict): Parameters for the step.

        Returns:
            Any: The result of the step execution.
        """
        logger.debug(f"Executing step {step_name} with params: {params}")

        step_methods = {
            "train": self.train,
            "validate": self.validate,
            "evaluate": self.evaluate,
            "retrain": self.retrain,
        }

        step_method = step_methods.get(step_name)
        if step_method:
            return step_method(**params)
        else:
            raise ValueError(f"Invalid step name: {step_name}")

    def _get_model_list(
        self, use_basemodel: bool, models_dir: Optional[str]
    ) -> Tuple[List[str], List[str]]:
        """Helper method to get the list of models to evaluate."""
        if use_basemodel:
            models_config_list = [self.pipeline_configuration.model_config_path]
            model_weights_list = [self.pipeline_configuration.weights_config_path]
        else:
            if not models_dir:
                raise ValueError("models_dir must be specified if not using basemodel")
            models_config_list = []
            model_weights_list = []
            for model_dir in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_dir)
                config_path = os.path.join(model_path, "config.yaml")
                weights_path = os.path.join(model_path, "best_weights.pth")
                if os.path.exists(config_path) and os.path.exists(weights_path):
                    models_config_list.append(config_path)
                    model_weights_list.append(weights_path)
                else:
                    logger.warning(
                        f"Missing config or weights in {model_path}, skipping..."
                    )
        return models_config_list, model_weights_list

    def _get_baseline_accuracy(
        self, model_config: str, model_weights: str
    ) -> float:
        """Helper method to compute baseline accuracy."""
        model = ModelBuilder().build(
            ModelTypes.JSC, config=model_config, weights_path=model_weights
        )
        accuracy, _ = self.pipeline_configuration.validator.validate(model)
        return accuracy

    def _gather_layer_info(
        self, model: Any, layers: List[str], bitwidth_weights_list: List[int]
    ) -> Dict[str, Any]:
        """Helper method to gather layer information."""
        layer_info = {}
        for layer, bitwidth in zip(layers, bitwidth_weights_list):
            params = get_weights_per_layer(model, layer)
            zero_params = get_weights_zero_value_per_layer(model, layer)
            max_usage_kb = (params * 32) / 8 / 1024
            curr_usage_kb = (params * bitwidth) / 8 / 1024
            layer_info[layer] = {
                "params": params,
                "zero_params": zero_params,
                "max_usage_kb": max_usage_kb,
                "curr_usage_kb": curr_usage_kb,
                "bitwidth": bitwidth,
            }
        return layer_info


def main():
    """Main function to execute the pipeline."""
    base_dir = "tmp_data/experiment_quant_first"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    base_model_config = os.path.join(
        parent_directory, "configs", "jsc", "quant_jsc_xl_updated_quant.yaml"
    )
    float_model_weight_path = os.path.join("weights", "jsc", "jsc_xl_weights.pth")

    processing_sequence = ["dense2", "dense3", "dense4", "dense1", "dense5"]

    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=base_model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

    config_manager = PipelineManager(
        dataset=dataset,
        validator=validator,
        model_config_path=base_model_config,
        weights_config_path=float_model_weight_path,
        layers=processing_sequence,
        results_path="report.json",
    )

    file_logger = logging.getLogger("file_logger")
    file_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("pipeline.log")
    file_handler.setLevel(logging.DEBUG)
    file_logger.addHandler(file_handler)

    pipeline = Pipeline(
        pipeline_configuration=config_manager,
        logger=file_logger,
        report_path="results.json",
    )

    # Define the steps and their corresponding parameters
    pipeline_steps = [
        {
            "name": "evaluate",
            "params": {
                "step_name": "0_evaluate_quantization",
                "eval_key": "weight_bit_width",
                "eval_range": list(range(16, 3, -1)),
                "weight_prune_first": False,
                "use_basemodel": True,
                "allowed_acc_drop": [50.0],
                "layers": processing_sequence,
                "models_dir": None,
                "save_dir": os.path.join(base_dir, "0_quantization"),
            },
        },
        {
            "name": "retrain",
            "params": {
                "step_name": "1_retrain_after_quantization",
                "lr": 0.00009349,
                "epochs": 5,
                "models_dir": os.path.join(base_dir, "0_quantization"),
            },
        },
        {
            "name": "evaluate",
            "params": {
                "step_name": "2_evaluate_pruning",
                "eval_key": "weight_sparse_eps",
                "eval_range": np.arange(0.0, 1.0, 0.01).tolist(),
                "weight_prune_first": False,
                "use_basemodel": False,
                "allowed_acc_drop": [4.0],
                "layers": processing_sequence,
                "models_dir": os.path.join(base_dir, "0_quantization"),
                "save_dir": os.path.join(base_dir, "1_pruning"),
            },
        },
        {
            "name": "retrain",
            "params": {
                "step_name": "3_retrain_after_pruning",
                "lr": 0.00009349,
                "epochs": 5,
                "models_dir": os.path.join(base_dir, "1_pruning"),
            },
        },
    ]

    # Run the pipeline
    pipeline.run(pipeline_steps)


if __name__ == "__main__":
    main()