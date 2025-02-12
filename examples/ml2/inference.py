import json
import shutil
from includes_ml2 import *
import numpy as np

base_model_config = f"/home/fry/Documents/repositories/synapselab/examples/ml2/float_model/config.yaml"
float_model_weight_path = f"/home/fry/Documents/repositories/synapselab/examples/ml2/float_model/best_weights.pth"

dataset = DatasetBuilder.build(DatasetTypes.JSC, config=base_model_config)
validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())

model = ModelBuilder().build(ModelTypes.JSC, config=base_model_config, weights_path=float_model_weight_path)

validator.validate(model)