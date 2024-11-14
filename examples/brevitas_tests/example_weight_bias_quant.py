import sys
import torch

import torch.nn as nn
import brevitas.nn as qnn
# Add the parent directory to the path so project modules can be imported
from pathlib import Path

current_script_path = Path(__file__).parent
parent_directory = current_script_path.parent.parent

sys.path.append(str(parent_directory))

from src.datasets.dataset_builder import DatasetBuilder
from src.models.model_builder import ModelBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner
from src.engine.config_manager import ConfigurationManager

class JscXl(nn.Module):
    
	def __init__(self):
		super(JscXl, self).__init__()
		self.dense1 = qnn.QuantLinear(in_features=16, out_features=128, bias=True)
		self.relu1 = nn.ReLU(inplace=True)
		self.dense2 = qnn.QuantLinear(in_features=128, out_features=64, bias=True)
		self.relu2 = nn.ReLU(inplace=True)
		self.dense3 = qnn.QuantLinear(in_features=64, out_features=64, bias=True)
		self.relu3 = nn.ReLU(inplace=True)
		self.dense4 = qnn.QuantLinear(in_features=64, out_features=64, bias=True)
		self.relu4 = nn.ReLU(inplace=True)
		self.dense5 = qnn.QuantLinear(in_features=64, out_features=5, bias=True)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.dense1(x)
		x = self.relu1(x)
		x = self.dense2(x)
		x = self.relu2(x)
		x = self.dense3(x)
		x = self.relu3(x)
		x = self.dense4(x)
		x = self.relu4(x)
		x = self.dense5(x)
		x = self.softmax(x)

		return x



model_config = "/home/mmecik/repositories/synapselab/configs/jsc/jsc_xl.yaml"

dataset = DatasetBuilder().build(DatasetTypes.JSC, config=model_config)
validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

model = JscXl()
model.load_state_dict(torch.load("/home/mmecik/repositories/synapselab/train/jsc_xl_floating_point/run_331/best_weights.pth"))

acc, _ = validator.validate(model)