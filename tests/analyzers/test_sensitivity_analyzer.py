import unittest
import traceback
import torch
import yaml

from src.models.model_builder import ModelBuilder, ModelTypes
from src.datasets.dataset_builder import DatasetBuilder, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer

from src.analyzer.sensitivity import SensitivityAnalyzer

class TestSensitivityAnalyzer(unittest.TestCase):
    def setUp(self):
        config_path = "configs/jsc/jsc_xl.yaml"
        model_builder = ModelBuilder()

        self.trainer = Trainer()

        self.model = model_builder.build(ModelTypes.JSC, config_path)
        self.dataset = DatasetBuilder.build(DatasetTypes.JSC, config_path)

        self.sensitivity_analyzer = SensitivityAnalyzer()

    """
    def test_accuracy(self):
        epochs = 10
        lr = 0.001

        baseline_accuracy, _ = self.validator.validate(self.model, None, self.dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
        print(f"Baseline accuracy: {baseline_accuracy}")

        self.trainer.train(self.model, None, self.dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(self.model.parameters(), lr), lr, epochs)
        accuracy, _ = self.validator.validate(self.model, None, self.dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

        if accuracy < baseline_accuracy:
            self.fail(f"Accuracy decreased from {baseline_accuracy} to {accuracy}")
    """

    def test_load_config_error(self):
        exception_raised = False
        try:
            config = self.sensitivity_analyzer._load_config("configs/jsc/jsc_xla.yaml")
        except:
            exception_raised = True

    def test_load_config_file_ok(self):
        exception_raised = False
        try:
            config = self.sensitivity_analyzer._load_config("configs/jsc/jsc_xl.yaml")
        except:
            exception_raised = True

        self.assertFalse(exception_raised)

    def test_load_config_ok(self):
        # Test if the config is loaded correctly
        with open("configs/jsc/jsc_xl.yaml", "r") as f:
            config = yaml.safe_load(f)
            self.assertIn(config['name'], "jsc_xl")

    def test_load_model_error(self):
        exception_raised = False
        try:
            model = self.sensitivity_analyzer._load_model(ModelTypes.JSC, "configs/jsc/jsc_xla.yaml")
        except:
            exception_raised = True

        self.assertTrue(exception_raised)

    def test_load_model_ok(self):
        exception_raised = False
        try:
            weights_path = "weights/jsc/jsc_xl_weights.pth"
            config_path = "configs/jsc/jsc_xl.yaml"
            model = self.sensitivity_analyzer._load_model(ModelTypes.JSC, config_path, weights_path)
        except:
            exception_raised = True
            traceback.print_exc()

        self.assertFalse(exception_raised)

    def test_set_config_param_error_layer(self):

        layer = "dense12"
        key = "output_bit_width"
        value = 3

        config_path = "configs/jsc/quant_jsc_xl.yaml"
        config = self.sensitivity_analyzer._load_config(config_path)

        exception_raised = False
        try:
            self.sensitivity_analyzer._set_layer_value_by_key(config, layer, key, value)
        except:
            exception_raised = True

        self.assertTrue(exception_raised)

    def test_set_config_param_error_key(self):

        layer = "dense1"
        key = "output_bit_widths"
        value = 3

        config_path = "configs/jsc/quant_jsc_xl.yaml"
        config = self.sensitivity_analyzer._load_config(config_path)

        exception_raised = False
        try:
            self.sensitivity_analyzer._set_layer_value_by_key(config, layer, key, value)
        except:
            exception_raised = True

        self.assertTrue(exception_raised)

    def test_set_config_param_ok(self):
        layer = "dense1"
        key = "output_bit_width"
        value = 3

        config_path = "configs/jsc/quant_jsc_xl.yaml"
        config = self.sensitivity_analyzer._load_config(config_path)

        # Set value
        self.sensitivity_analyzer._set_layer_value_by_key(config, layer, key, value)

        # Get value
        ret_val = self.sensitivity_analyzer._get_layer_value_by_key(config, layer, key)

        self.assertEqual(ret_val, value)

    def test_analyze_sensitivity_by_key_ok(self):

        config_path = "configs/jsc/quant_jsc_xl.yaml"
        weights_path = "weights/jsc/jsc_xl_weights.pth"

        config = self.sensitivity_analyzer._load_config(config_path)
        dataset = DatasetBuilder.build(DatasetTypes.JSC, config)
        validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

        layer = ["dense1", "dense2"]

        sensitivity = self.sensitivity_analyzer.analyze_sensitivity_by_key(ModelTypes.JSC, config, weights_path, validator, layer, "output_bit_width", [4,3,2], None, True)
        self.sensitivity_analyzer.plot_sensitivity(sensitivity, "output_bit_width.pdf")
        
        print(self.model)
        
    def tearDown(self):
        pass
