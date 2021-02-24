#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:41:38 2021

@author: swen
"""

import os
import time
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL -> hides tensor warning messages

from keras.backend import clear_session
from load_data import load_data
from run_new_model import run_new_model 
from save_model import save_model

x_train, x_test, train_lable, test_lable = load_data()              #Load data from folder  

for batchsize in [32, 64, 128, 256, 512]:
    start_time = time.time()
    model, history = run_new_model(x_train, x_test, train_lable, test_lable , 
                                      Batchnorm = False,
                                      Droprate = 0.5, 
                                      Batchsize = batchsize, 
                                      epochs = 20)
    end_time = time.time() - start_time
    print("--- %s seconds ---" % (end_time))

    save_model(model, history, "No_batchnorm_" + str(batchsize), end_time)

    del model, history                                              #.fit() blows up ram
    gc.collect()                                                    #memory management!
    clear_session()                                                 #memory management!
