#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Dataset, frame2numpy, Scenario,\
    Formal_Config, Formal_Configs, Vehicle
from deepgtav.client import Client

import argparse
import time
import scenic


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
    parser.add_argument('-sc', '--sc_file', default='examples/gta/platoonDaytime.sc')
    args = parser.parse_args()


    #name = "examples/gta/platoonDaytime.sc"
    name = args.sc_file

    startTime = time.time()
    # construct scenario
    scenario = scenic.scenarioFromFile(name)


    end = int(args.end)
    start = int(args.start)

    for idx in range(0, 1):

        scene, _ = scenario.generate()
        my_cfg = scenic.simulators.gta.interface.GTA.Config(scene)

        for car in my_cfg.vehicles:
            print(car)
        cfgs=[my_cfg]

        print("The object is :", my_cfg)
        with open("C:\Program Files (x86)\Steam\steamapps\common\Grand Theft Auto V\\kk\\config_" + args.index + ".txt", "w") as fw:

            fw.write(str(my_cfg.location) + "\n")
            fw.write(str(my_cfg.time) + "\n")
            fw.write(str(my_cfg.weather) + "\n")
            fw.write(str(my_cfg.vehicles) + "\n")
            fw.write(str(my_cfg.view_heading))

        client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9)

        client.sendMessage(Formal_Configs(cfgs))
