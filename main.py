from src.models.model_builder import ModelBuilder
from src.datasets.dataset_builder import DatasetBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner

import torch
import torchvision.models as models
import time
from colorama import Fore
from tqdm import tqdm


if __name__ == "__main__":
    modelbuilder = ModelBuilder()
    dataset = DatasetBuilder.build(DatasetTypes.MNIST, "configs/lenet5/config.yaml")
    #validator = Validator()

    #quant_model = modelbuilder.build(ModelTypes.VGG, "configs/vgg19/quant_config.yaml", preload_weights=True)
    model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/config.yaml", preload_weights=True)
    student_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/student_config.yaml", preload_weights=True)
    total_params_teacher = sum(p.numel() for p in model.parameters())
    total_params_student = sum(p.numel() for p in student_model.parameters())
    print(f"{Fore.GREEN} \n++++\nTotal Parameters of Teacher {total_params_teacher}\n+++{Fore.RESET}")
    print(f"{Fore.GREEN} \n++++\nTotal Parameters of Student {total_params_student}\n+++{Fore.RESET}")
    #Tuner.tune(quant_model, torch.optim.SGD, torch.nn.CrossEntropyLoss(), dataset, 5, 10, 10)

    trainer = Trainer()
    lr = 0.001
    #trainer.train(quant_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.SGD(quant_model.parameters()), 0.001)
    trainer.train_teacher_student(model, student_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), 3, T=2)
    #trainer.train(student_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr, 3)
    #validator.validate(quant_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
    #validator.validate(model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
