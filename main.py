from src.models.model_builder import ModelBuilder
from src.datasets.dataset_builder import DatasetBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner

import torch
import torchvision.models as models
import time
from tqdm import tqdm

if __name__ == "__main__":
    modelbuilder = ModelBuilder()
    dataset = DatasetBuilder.build(DatasetTypes.IMAGENET, "configs/vgg19/quant_config.yaml")
    validator = Validator()

    quant_model = modelbuilder.build(ModelTypes.VGG, "configs/vgg19/quant_config.yaml", preload_weights=True)
    model = modelbuilder.build(ModelTypes.VGG, "configs/vgg19/config.yaml", preload_weights=True)

    # Tuner.tune(quant_model, torch.optim.SGD, torch.nn.CrossEntropyLoss(), dataset, 5, 10, 10)

    # trainer = Trainer()
    # trainer.train(quant_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.SGD(quant_model.parameters()), 0.001)

    # validator.validate(quant_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
    validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
