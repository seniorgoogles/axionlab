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
    ### INITS ###
    modelbuilder = ModelBuilder()
    validator = Validator()
    trainer = Trainer()
    lr = 0.001
    epochs = 15

    ### PARENT ###
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/config.yaml")
    parent_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/config.yaml", preload_weights=True)
    total_params_teacher = sum(p.numel() for p in parent_model.parameters())
    
    ### CHILD ###
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/student_config.yaml")
    student_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/student_config.yaml", preload_weights=True)
    total_params_student = sum(p.numel() for p in student_model.parameters())

    ### CHILD KD ###
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/student_config.yaml")
    kd_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/student_config.yaml", preload_weights=True)

    ### CHILD QUANT ###
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/quant_student_config.yaml")
    quant_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/quant_student_config.yaml", preload_weights=True)
    total_params_quant = sum(p.numel() for p in student_model.parameters())

    ### TRAIN ###
    trainer.train(parent_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(parent_model.parameters(), lr), lr, epochs)
    trainer.train(student_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(student_model.parameters(), lr), lr, epochs)
    trainer.train_teacher_student(parent_model, kd_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(kd_model.parameters(), lr), epochs, T=2, teacherIsPreTrained=True)
    trainer.train(quant_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(quant_model.parameters(), lr), lr, epochs)
    
    #Tuner.tune(quant_model, torch.optim.SGD, torch.nn.CrossEntropyLoss(), dataset, 5, 10, 10)
    
    ### VALIDATE ###
    print(f"{Fore.GREEN} \n------------------------------------\nValidate Parent\n------------------------------------{Fore.RESET}")
    validator.validate(parent_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    print(f"{Fore.GREEN} \n------------------------------------\nValidate Child ONLY \n------------------------------------{Fore.RESET}")
    validator.validate(student_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    print(f"{Fore.GREEN} \n------------------------------------\nValidate Child WITH KD\n------------------------------------{Fore.RESET}")
    validator.validate(kd_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    print(f"{Fore.GREEN} \n------------------------------------\nValidate Child QUANT\n------------------------------------{Fore.RESET}")
    validator.validate(quant_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of Teacher {total_params_teacher}\n------------------------------------{Fore.RESET}")
    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of Student {total_params_student}\n------------------------------------{Fore.RESET}")
