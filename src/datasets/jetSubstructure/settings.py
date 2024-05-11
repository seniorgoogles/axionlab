import os

BATCH_SIZE = 1024
EPOCHS = 1000
LEARNING_RATE = 0.001
LR_SCHEDULER_STEP_SIZE = 1000

LSB_IN = -8
LSB_OUT = -8

# Default Parameters for JSC-M Lite
JSC_M_LITE_BITWIDTH = 3
JSC_M_LITE_MSB = 1
JSC_M_LITE_LSB = -1

# Default Parameters for JSC-XL
JSC_XL_BITWIDTH = 5
JSC_XL_MSB = 1
JSC_XL_LSB = -3

DYNAMIC_QUANTIZER_BIT_WIDTH = 3

# Sign bit included
STATIC_QUANTIZER_MSB = 1
STATIC_QUANTIZER_LSB = -1

# Bitwidths / MSBs and LSBs for hyperparameter tuning
BITWIDTH_TUNE_GRID = [(16, ), (12, ), (8, ), (6, ), (4, ), (3, ), (2, ), (1, )]
MSB_LSB_TUNE_GRID = [(2, -13), (2, -9), (2, -5), (1, -4), (1, -2), (1, -1), (0, -1), (0, 0)]

# Directories to save data
DATASET_ROOT_PATH = "jetSubstructure/JSC_Dataset"
DATA_ROOT_PATH = "data"
CONFIG_SAVE_DIR = os.path.join(DATA_ROOT_PATH, "configs")
MODEL_SAVE_DIR = os.path.join(DATA_ROOT_PATH, "checkpoints")
TRAIN_RESULTS_SAVE_DIR = os.path.join(DATA_ROOT_PATH, "training_results")
