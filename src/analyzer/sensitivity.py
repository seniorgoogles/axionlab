import os
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from matplotlib.backends.backend_pdf import PdfPages
from typing import Union, Optional

from src.datasets.dataset_builder import DatasetBuilder, DatasetTypes
from src.models.model_builder import ModelBuilder, ModelTypes
from src.engine.validator import Validator

from src.analyzer.models.sensitivity import Sensitivity

class SensitivityAnalyzer:

    def _load_config(self, config: Union[str, dict, None]) -> dict:
        """
        Loads the config object from a file if it is None.
        """
        if isinstance(config, str):
            config_path = config
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    return yaml.safe_load(file)
            else:
                raise FileNotFoundError(f"Config file {config_path} does not exist")
        elif config is None:
            raise ValueError("Config cannot be None")
        return config

    def _load_model(self, model_type: str, config: Union[str, dict, None], weights_path: str = None):
        """
        Loads the model from the config.
        """
        config = self._load_config(config)
        return ModelBuilder.build(model_type, config, preload_weights=False, weights_path=weights_path)

    def _set_layer_value_by_key(self, config: dict, layer: str, key: str, value, model_part: str = "backbone") -> dict:
        """
        Sets the value of a key in a specific layer configuration.
        """
        layer_config_list = config[model_part]
        for layer_config in layer_config_list:
            layer_name, layer_config_dict = layer_config[3], layer_config[4]
            if layer_name == layer:
                if key in layer_config_dict:
                    layer_config_dict[key] = value
                    return config
                else:
                    raise KeyError(f"Key {key} does not exist in layer {layer}")
        raise ValueError(f"Layer {layer} does not exist in config")

    def _get_layer_value_by_key(self, config: dict, layer: str, key: str, model_part: str = "backbone"):
        """
        Gets the value of a key in a specific layer configuration.
        """
        layer_config_list = config[model_part]
        for layer_config in layer_config_list:
            layer_name, layer_config_dict = layer_config[3], layer_config[4]
            if layer_name == layer:
                if key in layer_config_dict:
                    return layer_config_dict[key]
                else:
                    raise KeyError(f"Key {key} does not exist in layer {layer}")
        raise ValueError(f"Layer {layer} does not exist in config")

    def analyze_sensitivity_by_key(
        self,
        model_type: str,
        config: Union[str, dict],
        weights_path: str,
        validator: Validator,
        layer: Union[str, list],
        key: str,
        values: list,
        acc_drop_allowed: Optional[float] = None,
        reset_model: bool = False
    ):
        """
        Analyzes the sensitivity of a specific layer and key.
        """
        layers = [layer] if isinstance(layer, str) else layer
        default_config = self._load_config(config)
        default_model = self._load_model(model_type, default_config, weights_path)
        baseline_acc, baseline_loss = validator.validate(default_model)
        
        sensitivity_acc = Sensitivity(default_model, "Bit-Width Activations", "Accuracy")
        sensitivity_loss = Sensitivity(default_model, "Bit-Width Activations", "Loss")

        print(f"Baseline accuracy: {baseline_acc}")
        result_sensitivity = {}

        result_sensitivity = {
            "baseline_acc": baseline_acc,
            "baseline_loss": baseline_loss,
            "reset_model": reset_model,
            "acc_drop_allowed" : acc_drop_allowed,
            "sensitivity": {}
        }

        for layer in layers:
            tmp_config = deepcopy(default_config)
            result_sensitivity["sensitivity"][layer] = []
            for value in values:
                tmp_config = self._set_layer_value_by_key(tmp_config, layer=layer, key=key, value=value)
                model = self._load_model(model_type, tmp_config, weights_path)
                acc, loss = validator.validate(model)

                # Check if accuracy drop is set, check if accuracy is below the allowed accuracy
                if acc_drop_allowed != None:
                    if acc < (baseline_acc - acc_drop_allowed):
                        print(f"Accuracy is lower than allowed drop")
                        break

                sensitivity_acc.add(value, acc)
                sensitivity_loss.add(value, loss)
            if reset_model:

                tmp_config = deepcopy(default_config)

        return [sensitivity_acc, sensitivity_loss]

    def plot_sensitivity(self, model_config, data: list, pdf_path="senstivity_analysis.pdf"):
        
        pdf_pages = PdfPages(pdf_path)

        """
        pdf_pages = PdfPages(pdf_path)
        
        # Title page
        plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
        plt.text(0.5, 0.5, 'Sensitivity List', horizontalalignment='center', verticalalignment='center', fontsize=20)
        plt.axis('off')
        pdf_pages.savefig()
        plt.close()

        # Extract keys dynamically
        sample_metrics = next(iter(data['sensitivity'].values()))[0]
        metric_keys = list(sample_metrics.keys())
        metric_keys.remove('output_bit_width')
        """
        """
        # Overview plot for all layers
        fig, axs = plt.subplots(1, 2, figsize=(16.54, 11.69))  # Two A4 pages side by side
        for metric in metric_keys:
            for layer, metrics in data['sensitivity'].items():
                bit_widths = np.array([item['output_bit_width'] for item in metrics])
                values = np.array([item[metric] for item in metrics])
                
                if metric == 'accuracy':
                    baseline = baseline_acc
                    ylabel = 'Accuracy'
                    ax = axs[0]
                elif metric == 'loss':
                    baseline = baseline_loss
                    ylabel = 'Loss'
                    ax = axs[1]
                else:
                    baseline = None
                    ylabel = metric.capitalize()
                
                indices = np.argsort(bit_widths)[::-1]  # Reverse sorting to have highest bit width on the left
                bit_widths = bit_widths[indices]
                values = values[indices]
                
                ax.plot(bit_widths, values, marker='o', label=f'{layer} {ylabel}')
                if baseline is not None:
                    ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {ylabel}')
                ax.set_xlabel('Output Bit Width')
                ax.set_ylabel(ylabel)
                ax.set_title(f'Sensitivity Analysis of {ylabel} for All Layers')
                ax.invert_xaxis()  # Inverting x-axis
                ax.set_xticks(np.arange(min(bit_widths), max(bit_widths) + 1, 1))  # Set precision to 1
                ax.legend()
                ax.grid(True)
        pdf_pages.savefig(fig)
        plt.close()

        # Plotting each layer separately
        for layer, metrics in data['sensitivity'].items():
            bit_widths = np.array([item['output_bit_width'] for item in metrics])
            
            # Prepare plots
            fig, axs = plt.subplots(1, 2, figsize=(16.54, 11.69))  # Two A4 pages side by side
            
            for i, metric in enumerate(metric_keys):
                values = np.array([item[metric] for item in metrics])
                
                if metric == 'accuracy':
                    baseline = baseline_acc
                    ylabel = 'Accuracy'
                    ax = axs[0]
                elif metric == 'loss':
                    baseline = baseline_loss
                    ylabel = 'Loss'
                    ax = axs[1]
                else:
                    baseline = None
                    ylabel = metric.capitalize()

                indices = np.argsort(bit_widths)[::-1]  # Reverse sorting to have highest bit width on the left
                bit_widths_sorted = bit_widths[indices]
                values_sorted = values[indices]

                ax.plot(bit_widths_sorted, values_sorted, marker='o', label=f'{layer} {ylabel}')
                if baseline is not None:
                    ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline {ylabel}')
                ax.set_xlabel('Output Bit Width')
                ax.set_ylabel(ylabel)
                ax.set_title(f'Sensitivity Analysis of {ylabel} for {layer}')
                ax.invert_xaxis()  # Inverting x-axis
                ax.set_xticks(np.arange(min(bit_widths_sorted), max(bit_widths_sorted) + 1, 1))  # Set precision to 1
                ax.legend()
                ax.grid(True)
            
            # Save the figure to the PDF
            pdf_pages.savefig(fig)
            plt.close()
        
        pdf_pages.close()
        """