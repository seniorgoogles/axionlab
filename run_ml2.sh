#!/bin/bash

# Copy and overwrite the files
cp -f ml2_jsc_* ../NNController/

# Go to the directory 
cd ../NNController/

# Run the program
./bin/NNController performSanityCheckJSON=1 outputDirectory=/home/mmecik/tmp_out_ml2 filename=„NNController.vhdl“ validateValidationData=0 verbose=1 pathToConfig=data/ml2_jsc_config.json pathToData=data/ml2_jsc_weights.json