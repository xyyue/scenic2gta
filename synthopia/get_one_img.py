#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Dataset, frame2numpy, Scenario,\
    Formal_Config, Formal_Configs, Vehicle
from deepgtav.client import Client

from random import randint
from scenarios import Map
import test_scenarios

import argparse
import time
import cv2
import scenic

from translator import *
#from example import get_one_example
# Stores a dataset file with data coming from DeepGTAV
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-d', '--dataset_path', default='dataset.pz', help='Place to store the dataset')
    parser.add_argument('-i', '--index', default='5')
    parser.add_argument('-m', '--l_r', default='0')
    parser.add_argument('-n', '--f_b', default='10')
    parser.add_argument('-end', '--end', default='100')
    parser.add_argument('-start', '--start', default='0')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port. 
    # If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
    #client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9)
    
    # Configures the information that we want DeepGTAV to generate and send to us. 
    # See deepgtav/messages.py to see what options are supported
    # dataset = Dataset(rate=30, frame=[320,160], throttle=True, brake=True, steering=True, vehicles=True, peds=True, reward=[15.0, 0.0], direction=None, speed=True, yawRate=True, location=True, time=True)
    # Send the Start request to DeepGTAV.


    car1 = Vehicle(model="VOLTIC", color=[0,225,225], location_offset=[0,10,0], heading=180)
    car2 = Vehicle(model="BLISTA", color=[225,0,225], location_offset=[0, 20,0], heading=45)
    cfg1 = Formal_Config(location=[127.4, -1307.7, 29.2], time=[9,20], weather="EXTRASUNNY", vehicles=[car1, car2], view_heading=180.0)
    cfg2 = Formal_Config(location=[1747.0, 3273.7, 41.1], time=[17,20], weather="XMAS", vehicles=[car1, car2], view_heading=0.0)
    cfg3 = Formal_Config(location=[2354.0, 1830.3, 101.1], time=[12,30], weather="RAIN", vehicles=[car2], view_heading=90.0)
    
    car6 = Vehicle(model="BLISTA", color=[128,225,225], location_offset=[int(args.l_r),int(args.f_b),0], heading=0)
    
    
    car7 = Vehicle(model="BLISTA", color=[-780,0,0], location_offset=[5,10,0], heading=90)
    car7 = Vehicle(model="BALLER", color=[0,225,225], location_offset=[0,10,0], heading=0)
    car8 = Vehicle(model="BLISTA", color=[204,0,0], location_offset=[-0,10,0], heading=0)
    car9 = Vehicle(model="BALLER", color=[0,0,0], location_offset=[3,8,0], heading=200)
    cfg666 = Formal_Config(location=[-583.4598, -315.3495, 60], time=[12,20], weather="EXTRASUNNY", vehicles=[car8, car8], view_heading=297.1754)


    # strip extension off filename if necessary
    name = "scenarios/platoonDaytime"

    startTime = time.time()
    # construct scenario
    scenario = scenic.scenarioFromFile(name)


    end = int(args.end)
    start = int(args.start)

    for idx in range(0, 1):

        scene, _ = scenario.generate()
        cfg666 = scenic.simulators.gta.interface.GTA.Config(scene)

        for car in cfg666.vehicles:
            print(car)
        cfgs=[cfg666]

        print("The object is :", cfg666)
        with open("C:\Program Files (x86)\Steam\steamapps\common\Grand Theft Auto V\\kk\\config_" + args.index + ".txt", "w") as fw:
        # with open(args.index + ".txt", "w") as fw:
            # fw.write(str(cfg666))
            fw.write(str(cfg666.location) + "\n")
            fw.write(str(cfg666.time) + "\n")
            fw.write(str(cfg666.weather) + "\n")
            fw.write(str(cfg666.vehicles) + "\n")
            fw.write(str(cfg666.view_heading))

        #if idx == start:
        client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9)

        client.sendMessage(Formal_Configs(cfgs))
