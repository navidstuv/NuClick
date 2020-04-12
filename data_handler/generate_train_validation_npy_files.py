# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:05:45 2020

@author: Mosi
"""
from data_handler.npyDataOps import infosToNumpyData
from config import config

# working on train folder
dataPath = config.train_data_path
infosToNumpyData(dataPath)

# working on validation dataset, if avaiable
if not config.valid_data_path == None:
    infosToNumpyData(config.valid_data_path)
