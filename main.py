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


if __name__ == "__main__":
    '''
    quant_linear = QuantLinear(2, 4, weight_quant=LearnedBitWidthQuantizer, bias=False)
    print(f"Weight QuantTensor:\n {quant_linear.quant_weight()}")
    '''
    
    ################################################### INITS ###################################################
    modelbuilder = ModelBuilder()
    validator = Validator()
    trainer = Trainer()
    lr = 0.001
    epochs = 50
    custom_quant_config = "configs/lenet5/config_custom_quant.yaml"

    ### PARENT ###
    parent_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/config.yaml", preload_weights=True)
    total_params_teacher = sum(p.numel() for p in parent_model.parameters())

    quant_parent_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/quant_config.yaml", preload_weights=True)
    total_params_quant_teacher = sum(p.numel() for p in quant_parent_model.parameters())
    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of KD START CUT  {total_params_quant_teacher}\n------------------------------------{Fore.RESET}")

    ### CHILD ###
    student_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/student_config.yaml", preload_weights=True)
    total_params_student = sum(p.numel() for p in student_model.parameters())

    ### CHILD END REDUCTION ###
    end_student = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/end_student_config.yaml", preload_weights=True)
    total_params_END_student = sum(p.numel() for p in end_student.parameters())
   
    ### CHILD START REDUCTION ###
    start_student = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/start_student_config.yaml", preload_weights=True)
    total_params_START_student = sum(p.numel() for p in start_student.parameters())
  
    ### CHILD KD ###
    kd_model = modelbuilder.build(ModelTypes.LENET, "configs/lenet5/student_config.yaml", preload_weights=True)

    ### REF QUANT ###
    refQuant = modelbuilder.build(ModelTypes.LENET, custom_quant_config, preload_weights=True)
    total_params_quant = sum(p.numel() for p in refQuant.parameters())

    ### CHILD QKD ###
    qkd_model = modelbuilder.build(ModelTypes.LENET, custom_quant_config, preload_weights=True)

    ### DCQ QUANT ###
    dcq_model = modelbuilder.build(ModelTypes.LENET, custom_quant_config, preload_weights=True)

    ################################################### TRAIN ###################################################
    print(f"{Fore.MAGENTA} \n------------------------------------\nTraining Parent\n------------------------------------{Fore.RESET}")
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/config.yaml")
    trainer.train(parent_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(parent_model.parameters(), lr), lr, epochs)
    
    print(f"{Fore.MAGENTA} \n------------------------------------\nTraining DCQ WITH PARENT \n------------------------------------{Fore.RESET}")
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/quant_config.yaml")
    trainer.train_dcq(parent_model, quant_parent_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(quant_parent_model.parameters(), lr), epochs, 2,teacherIsPreTrained=True)
    
    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining CHILD ONLY\n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/student_config.yaml")
    # trainer.train(student_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(student_model.parameters(), lr), lr, epochs)
 
    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining KD \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/student_config.yaml")
    # trainer.train_teacher_student(parent_model, kd_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(kd_model.parameters(), lr), epochs, T=2, teacherIsPreTrained=True)

    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining CHILD END KD \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/end_student_config.yaml")
    # trainer.train_teacher_student(parent_model, end_student, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(end_student.parameters(), lr), epochs, T=2, teacherIsPreTrained=True)

    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining CHILD STATZ KD \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/end_student_config.yaml")
    # trainer.train_teacher_student(parent_model, start_student, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(start_student.parameters(), lr), epochs, T=2, teacherIsPreTrained=True)


    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining QUANT CHILD ONLY\n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, custom_quant_config)
    # trainer.train(refQuant, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(refQuant.parameters(), lr), lr, epochs)
    
    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining QUANT KD\n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, custom_quant_config)
    # trainer.train_teacher_student(parent_model, qkd_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(qkd_model.parameters(), lr), epochs, T=2, teacherIsPreTrained=True)
    
    # print(f"{Fore.MAGENTA} \n------------------------------------\nTraining DCQ\n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, custom_quant_config)
    # trainer.train_dcq(parent_model, dcq_model, None, dataset, torch.nn.CrossEntropyLoss(), torch.optim.Adam(dcq_model.parameters(), lr), 15, 2,teacherIsPreTrained=True)
    
    #Tuner.tune(quant_model, torch.optim.SGD, torch.nn.CrossEntropyLoss(), dataset, 5, 10, 10)
    
    ################################################### VALIDATE ###################################################
    print(f"{Fore.GREEN} \n------------------------------------\nValidate Parent\n------------------------------------{Fore.RESET}")
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/config.yaml")
    validator.validate(parent_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    print(f"{Fore.GREEN} \n------------------------------------\nValidate DCQ Parent \n------------------------------------{Fore.RESET}")
    dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/quant_config.yaml")
    validator.validate(quant_parent_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate Child ONLY \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/student_config.yaml")
    # validator.validate(student_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())
   
    # print(f"{Fore.GREEN} \n------------------------------------\nValidate END KD \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/end_student_config.yaml")
    # validator.validate(end_student, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate START KD \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/start_student_config.yaml")
    # validator.validate(start_student, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate KD \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, "configs/lenet5/student_config.yaml")
    # validator.validate(kd_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate QUANT Child ONLY \n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, custom_quant_config)
    # validator.validate(refQuant, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate KD QUANT\n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, custom_quant_config)
    # validator.validate(qkd_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    # print(f"{Fore.GREEN} \n------------------------------------\nValidate DCQ CHILD\n------------------------------------{Fore.RESET}")
    # dataset = DatasetBuilder.build(DatasetTypes.FASHION_MNIST, custom_quant_config)
    # validator.validate(dcq_model, None, dataset.get_test_loader(), torch.nn.CrossEntropyLoss())

    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of Teacher {total_params_teacher}\n------------------------------------{Fore.RESET}")
    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of Student {total_params_student}\n------------------------------------{Fore.RESET}")
    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of Quant Student {total_params_quant}\n------------------------------------{Fore.RESET}")
    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of KD END CUT {total_params_END_student}\n------------------------------------{Fore.RESET}")
    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of KD START CUT  {total_params_START_student}\n------------------------------------{Fore.RESET}")
    print(f"{Fore.GREEN} \n------------------------------------\nTotal Parameters of KD START CUT  {total_params_quant_teacher}\n------------------------------------{Fore.RESET}")
