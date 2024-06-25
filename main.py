from src.models.model_builder import ModelBuilder
from src.datasets.dataset_builder import DatasetBuilder
from src.core.inject.enum import ModelTypes, DatasetTypes
from src.engine.validator import Validator
from src.engine.trainer import Trainer
from src.engine.tuner import Tuner
from src.quantizer.learned_bitwidth_quantizer import LearnedBitWidthQuantizer

from brevitas.nn import QuantLinear

import torch
import torchvision.models as models
import time
from colorama import Fore
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    '''
    quant_linear = QuantLinear(2, 4, weight_quant=LearnedBitWidthQuantizer, bias=False)
    print(f"Weight QuantTensor:\n {quant_linear.quant_weight()}")
    '''
    
    ### INITS ###
    modelbuilder = ModelBuilder()
    validator = Validator()
    trainer = Trainer()
    lr = 0.001
    epochs = 25

    #parent_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/config_sparse_quant.yaml", preload_weights=True)
    #print(parent_model)
    ### PARENT ###
    #dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/config_custom_quant.yaml")
    #trainer.train(parent_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(parent_model.parameters(), lr), lr, epochs)
    #parent_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/config_custom_quant.yaml", preload_weights=True)
    #total_params_teacher = sum(p.numel() for p in parent_model.parameters())
    
    ### CHILD ###
    #dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/student_config.yaml")
    #student_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/student_config.yaml", preload_weights=True)
    #total_params_student = sum(p.numel() for p in student_model.parameters())

    ### REF QUANT ###
    #dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/quant_student_config.yaml")
    #refQuant = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/quant_student_config.yaml", preload_weights=True)

    ### CHILD KD ###
    #dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/quant_student_config.yaml")
    #qkd_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/quant_student_config.yaml", preload_weights=True)

    ### CHILD QUANT ###
    #dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/quant_student_config.yaml")
    #quant_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/quant_student_config.yaml", preload_weights=True)

    ### DCQ QUANT ###
    #dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/config_custom_quant.yaml")
    #dcq_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/config_custom_quant.yaml", preload_weights=True)

    ### TRAIN ###
    #print(f"{Fore.MAGENTA} \n------------------------------------\nTraining Parent\n------------------------------------{Fore.RESET}")
    #trainer.train(parent_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(parent_model.parameters(), lr), lr, epochs)
    #trainer.train(refQuant, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(refQuant.parameters(), lr), lr, epochs)
    # trainer.train(student_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(student_model.parameters(), lr), lr, epochs)
    # trainer.train(quant_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(quant_model.parameters(), lr), lr, epochs)
    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining QUANT KD\n------------------------------------{Fore.RESET}")
    # trainer.train_teacher_student(parent_model, qkd_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(qkd_model.parameters(), lr), epochs, T=2, teacherIsPreTrained=True)
    
    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining DCQ\n------------------------------------{Fore.RESET}")
    # trainer.train_dcq(parent_model, dcq_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(dcq_model.parameters(), lr), epochs, 2,teacherIsPreTrained=True)
    # #Tuner.tune(quant_model, torch.optim.SGD, torch.nn.CrossEntropyLoss(), dataset, 5, 10, 10)
    
    # ### VALIDATE ###
    # print(f"{Fore.GREEN} \n------------------------------------\nValidate Parent\n------------------------------------{Fore.RESET}")
    # validator.validate(parent_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate Child ONLY \n------------------------------------{Fore.RESET}")
    # validator.validate(student_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate Child QUANT\n------------------------------------{Fore.RESET}")
    # validator.validate(quant_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    #print(f"{Fore.GREEN} \n------------------------------------\nValidate REF QUANT\n------------------------------------{Fore.RESET}")
    #validator.validate(refQuant, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
    
    # print(f"{Fore.GREEN} \n------------------------------------\nValidate KD QUANT\n------------------------------------{Fore.RESET}")
    # validator.validate(qkd_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate DCQ CHILD\n------------------------------------{Fore.RESET}")
    # validator.validate(dcq_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of Teacher {total_params_teacher}\n------------------------------------{Fore.RESET}")
    # print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of Student {total_params_student}\n------------------------------------{Fore.RESET}")

    #resnet18 = models.resnet18(pretrained=True)
    #validator.validate(resnet18, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
    dataset = DatasetBuilder.build(DatasetTypes.JSC, "configs/jsc/jsc_m_lite_floating_point.yaml")
    #dataset_mnist = DatasetBuilder.build(DatasetTypes.MNIST, "configs/hdr/hdr_5l.yaml")
    model = modelbuilder.build(ModelTypes.JSC, "configs/jsc/jsc_m_lite_floating_point.yaml", preload_weights=True)
    #model_hrd = modelbuilder.build(ModelTypes.HDR, "configs/hdr/hdr_5l.yaml", preload_weights=True)
    print(model)
    trainer.train(model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr), lr, epochs)
