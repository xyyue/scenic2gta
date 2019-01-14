from gta import *

ego = Car
c2 = Car visible
c3 = Car at c2 offset by (-10, 1) @ 0
require (relative heading of c3 from c2) >= 150 deg