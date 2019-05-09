#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Dataset, frame2numpy, Scenario,\
    Formal_Config, Formal_Configs, Vehicle
from deepgtav.client import Client

import argparse
import time

from translator import *
# import importlib
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


    name = "scenarios/platoonDaytime"

    # import module
    startTime = time.time()
    module = importlib.import_module(name)
    # construct scenario
    scenario = constructScenarioFrom(module)

    end = int(args.end)
    start = int(args.start)

    for idx in range(0, 1):
        scene = scenario.generate()
        for obj in scene.cars:
            obj.position = obj.position.toVector()      # TODO remove hack

        my_cfg = scene.toSimulatorConfig()
        print("camera location: ", my_cfg.location)
        for car in my_cfg.vehicles:
            print(car)
        cfgs=[my_cfg]

        print("The object is :", my_cfg)
        with open("C:\Program Files (x86)\Steam\steamapps\common\Grand Theft Auto V\\kk\\config_" + args.index + ".txt", "w") as fw:
        # with open(args.index + ".txt", "w") as fw:
            # fw.write(str(my_cfg))
            fw.write(str(my_cfg.location) + "\n")
            fw.write(str(my_cfg.time) + "\n")
            fw.write(str(my_cfg.weather) + "\n")
            fw.write(str(my_cfg.vehicles) + "\n")
            fw.write(str(my_cfg.view_heading))

        client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9)
        client.sendMessage(Formal_Configs(cfgs))