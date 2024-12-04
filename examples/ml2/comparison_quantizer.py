import shutil

from matplotlib import pyplot as plt
from includes_ml2 import *


if __name__ == "__main__":
    base_model_config = f"{parent_directory}/weights/jsc/config.yaml"
    float_model_weight_path = f"{parent_directory}/weights/jsc/best_weights.pth"
    
    processing_sequence = ["dense2", "dense3", "dense4", "dense1", "dense5"]
    
    float_model = ModelBuilder().build(ModelTypes.JSC, config=base_model_config, weights_path=float_model_weight_path)

    dataset = DatasetBuilder().build(DatasetTypes.JSC, config=base_model_config)
    validator = Validator(torch.nn.CrossEntropyLoss(), dataset.get_test_loader())