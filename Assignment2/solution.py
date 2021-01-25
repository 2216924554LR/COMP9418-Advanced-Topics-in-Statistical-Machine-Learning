
'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: Rui Li    zID: z5224057

Name: Zhonglin Shi     zID: z5236346
'''

# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

# Allowed libraries 
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

###################################
# Code stub
# 
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
# 


# this global state variable demonstrates how to keep track of information over multiple 
# calls to get_action 

sensor_list =  ['reliable_sensor1', 'reliable_sensor2', 'reliable_sensor3', 
               'reliable_sensor4', 'unreliable_sensor1', 'unreliable_sensor2', 
               'unreliable_sensor3', 'unreliable_sensor4']
    
actions_dict = {'lights1': 'off', 'lights2': 'off', 'lights3': 'off', 'lights4': 'off', 'lights5': 'off', 'lights6': 'off', 
                'lights7': 'off', 'lights8': 'off', 'lights9': 'off', 'lights10': 'off', 'lights11': 'off', 'lights12': 'off', 
                'lights13': 'off', 'lights14': 'off', 'lights15': 'off', 'lights16': 'off', 'lights17': 'off', 'lights18': 'off', 
                'lights19': 'off', 'lights20': 'off', 'lights21': 'off', 'lights22': 'off', 'lights23': 'off', 'lights24': 'off', 
                'lights25': 'off', 'lights26': 'off', 'lights27': 'off', 'lights28': 'off', 'lights29': 'off', 'lights30': 'off', 
                'lights31': 'off', 'lights32': 'off', 'lights33': 'off', 'lights34': 'off', 'lights35':'off'}


# record last data sensor data
last_sensor = {'reliable_sensor1': 'no motion', 'reliable_sensor2': 'no motion', 'reliable_sensor3': 'no motion', 
               'reliable_sensor4': 'no motion', 'unreliable_sensor1': 'no motion', 'unreliable_sensor2': 'no motion', 
               'unreliable_sensor3': 'no motion', 'unreliable_sensor4': 'no motion', 'robot1': "('r1', 0)", 
               'robot2': "('r19', 0)", 'door_sensor1': 0, 'door_sensor2': 0, 'door_sensor3': 0, 'door_sensor4': 0, 
               'time': datetime.time(8, 0, 0), 'electricity_price': 1}

df = pd.read_csv('data.csv')

room_list =  ['r1', 'r2', 'r3', 'r4',
       'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
       'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25',
       'r26', 'r27', 'r28', 'r29', 'r30', 'r31', 'r32', 'r33', 'r34', 'r35']

sensor_table = df[sensor_list]
sensor_table = sensor_table.where(sensor_table=='motion', 0)
sensor_table = sensor_table.where(sensor_table==0, 1)
X = sensor_table

# build models for each room
model_list = []
for r in room_list:
    y = df[r]
    df[r] = df[r].where(df[r]==0, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    gnb = BernoulliNB()
    gnb.fit(X_train, y_train)
    #print(accuracy_score(y_test, gnb.predict(X_test)))
    model_list.append(gnb)


# params = pd.read_csv(...)

def sensor_num(sensor_data):
    if sensor_data['reliable_sensor1'] == 'motion':
        temp = '1'
    else:
        temp = '0'
    for j in sensor_list[1:]:
        if sensor_data[j] == 'motion':
            temp = temp + '1'
        else:
            temp = temp + '0'
    return int(temp)

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global actions_dict
    global last_senor
    
    
    # if sensor_data = Nan, use last data as current data
    for x in sensor_data:
        if  sensor_data[x] == None:
            sensor_data[x] = last_sensor[x]

    last_senor = sensor_data
    
    
    temp = []
    for j in sensor_list:
        if sensor_data[j] == 'motion':
            temp.append(1)
        else:
            temp.append(0)


    for m in range(len(model_list)):
        lights_num = 'lights' + str(m+1)
        p = model_list[m].predict_proba([temp])[0]
        #predict_list.append(p)
        
        if 4*p[1] > p[0]:
            actions_dict[lights_num] = 'on'
        else:
            actions_dict[lights_num] = 'off'
            
    
    # robot
    ro1 = eval(sensor_data['robot1'])
    
    if ro1[0] == 'outside':
        pass
    else:
        ro1_num = int(ro1[0][1:])
        lights_num = 'lights' + str(ro1_num)
        if ro1[0].startswith('r'):
            if ro1[1] > 0:
                actions_dict[lights_num] = 'on'
            else:
                actions_dict[lights_num] = 'off'
    
    
    ro2 = eval(sensor_data['robot2'])
    if ro2[0] == 'outside':
        pass
    else:
        ro2_num = int(ro2[0][1:])
        lights_num = 'lights' + str(ro2_num)
        if ro2[0].startswith('r'):
            if ro2[1] > 0:
                actions_dict[lights_num] = 'on'
            else:
                actions_dict[lights_num] = 'off'
                
    
    return actions_dict




