#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:34:07 2020

@author: evkikum
"""


import pandas as pd
import numpy as np
import os

os.chdir(r"/home/evkikum/Desktop/Data Science/Python/Practice/Capstone Project/text_topics")


## THE BELOW CODE HELPS IN READING ALL THE MULTI FILES DATA AND PUSH TO dictionary files.

filenames = os.listdir(r"/home/evkikum/Desktop/Data Science/Python/Practice/Capstone Project/text_topics")

files = {} 
 
for filename in filenames: 
    with open(filename, "r") as file: 
        if filename in files: 
            continue 
        files[filename] = file.read() 

'''
for filename, text in files.items(): 
    print(filename) 
    print("=" * 80) 
    print(text) 
''' 


