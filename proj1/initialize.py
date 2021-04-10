import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta


def read_velodyne_points(path):
    velodyne_data_filles = os.listdir(path)
    data_points = []
    for data_file in velodyne_data_filles:
        f = open(path + '\\' + data_file, "rb")
        data = np.fromfile(f, np.float32)
        data.shape = (-1, 4)
        data_points.append(data)
        f.close()
        
    return data_points


def initialize(data_path):
    oxt_data_keys = ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 'vf', 'vl', 'vu', 'ax', 'ay', 'af', 'al', 'au', 'wx', 'wy', 'wz',
                    'wf', 'wl', 'wu', 'pos_accuracy', 'vel_accuracy', 'navstat', 'numsats', 'posmode', 'velmode', 'orimode']
    # reading oxts
    oxts_times_file = data_path + '\\oxts\\timestamps.txt'
    oxts_data_path = data_path + '\\oxts\\data\\'
    oxts_times = []
    with open(oxts_times_file) as f:
        line = f.readline()[0:-4]
        oxts_times.append(datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f'))
        while line:
            line = f.readline()[0:-4]
            if line:
                oxts_times.append(datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f'))
                
        
    oxts_data_points = pd.DataFrame()
    data_dict = {}
    for data_file in os.listdir(oxts_data_path):
        with open(oxts_data_path + '\\' + data_file) as f:
            data_line = f.readline()
            broken_line = data_line.split(' ')
            
            for key in oxt_data_keys:
                if key not in data_dict.keys():
                    data_dict[key] = [broken_line.pop(0)]
                    
                else:
                    data_dict[key].append(broken_line.pop(0))
        
    data = pd.DataFrame(index=oxts_times, data=data_dict)
    data['velodyne_points'] = read_velodyne_points(data_path  + '\\velodyne_points\\data')

    return data