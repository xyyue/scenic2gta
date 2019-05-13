# scenic2gta

*This repo implements an interface to GTAV for [SCENIC](https://github.com/BerkeleyLearnVerify/Scenic) language. The GTAV plugin and communication parts are based on *[DeepGTAV](https://github.com/aitorzip/DeepGTAV)* and *[VPilot](https://github.com/aitorzip/VPilot)*.*

## Setting

1. Follow the necessary setup steps in *[DeepGTAV](https://github.com/aitorzip/DeepGTAV)*.
2. Run *setup.py* to do additional setups of the environment. 
3. Copy *.bins into the folder where the GTAV binay is.
4. Run GTAV and get into the game. 
5. Press *F4* to get the menue; use the numpad to navigate the menu (8 - Up, 2 - Down, 5 - Select): select “CAR” - “QUICK START”; press F4 again to hide the menu. 
6. Open a terminal and run:
```bash
bash get_images.sh examples/gta/platoonDaytime.sc 25
```
to get 25 images of scenic program platoonDaytime, then you'll get the data stored in the *kk* subfolder under the GTAV main directory. 
7. Run parse_data.py to get the parsed data. 
