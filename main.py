#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:41:38 2021

@author: swen
"""

import os
import time
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL -> hides tensor warning messages

from load_data import load_data
from run_new_model import run_new_model 
from save_model import save_model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch", type=bool, default = True,
    help="Should batch normalisation be used?")
ap.add_argument("-s", "--batchsize", type=int, default = 32,
    help="Batchsize?")
ap.add_argument("-d", "--dropout", type=float, default= 0.5,
    help="Dropout rate hidden layers")
ap.add_argument("-o", "--output", type=str, default = "results",
    help="Folder where results should be stored")
ap.add_argument("-e", "--epochs", type=int, default = 20,
    help="How many epochs?")
args = vars(ap.parse_args())

x_train, x_test, train_lable, test_lable = load_data()              #Load data from folder  

start_time = time.time()
model, history = run_new_model(x_train, x_test, train_lable, test_lable , 
                              Batchnorm = args['batch'],
                              Droprate = args['dropout'], 
                              Batchsize = args['batchsize'], 
                              epochs = args['epochs'])
end_time = time.time() - start_time
print("--- %s seconds ---" % (end_time))

save_model(model, history, args['output'], end_time)