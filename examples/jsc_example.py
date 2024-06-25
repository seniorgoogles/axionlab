from src.models.model_builder import ModelBuilder
from src.datasets.dataset_builder import DatasetBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner
from src.quantizer.learned_bitwidth_quantizer import LearnedBitWidthQuantizer

import os
import torch
from pathlib import Path

if __name__ == "__main__":
    # Set the project root directory
    os.environ['PROJECT_ROOT'] = str(Path.cwd().parent)

    modelbuilder = ModelBuilder()
    validator = Validator()
    trainer = Trainer()
    lr = 0.001
    epochs = 25

    dataset = DatasetBuilder.build(DatasetTypes.JSC, "../configs/jsc/jsc_xl.yaml")

    configs = [
        "../configs/jsc/jsc_2l.yaml",
        "../configs/jsc/jsc_5l.yaml",
        "../configs/jsc/jsc_lite.yaml",
        "../configs/jsc/jsc_m_lite_floating_point.yaml",
        "../configs/jsc/jsc_xl.yaml",
        "../configs/jsc/jsc_xl_floating_point.yaml"
    ]

    for config in configs:
        try:
            model = modelbuilder.build(ModelTypes.JSC, config, preload_weights=True)
            trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr,epochs)
        except Exception as e:
            print(f"Error: {e}")
            continue