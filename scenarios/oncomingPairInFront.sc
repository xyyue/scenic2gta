from gta import *

ego = Car
c2 = Car offset by (-10, 10) @ (20, 40), with viewAngle 30 deg
c3 = Car at c2 offset by (-10, 10) @ 0, with viewAngle 30 deg
require c2 can see ego
require c3 can see ego