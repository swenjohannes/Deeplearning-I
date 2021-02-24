#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:03:05 2021

@author: swen
"""

from pandas import DataFrame

def save_model(model, history, filename, time, memory):
    filename = 'Models/' + filename                #Models are stored in a folder
    model.save(filename)                            #save the model
    df = DataFrame.from_dict(history)
    df['Model'] = filename
    df['Time'] = time
    df['Memory'] = memory
    df.to_csv(filename + '/results.csv' ,index=False) 