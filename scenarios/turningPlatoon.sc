from gta import *

ego = Car
c2 = Car visible
platoon = createPlatoonAt(c2, 5, dist=(2, 8))
start = platoon[0]
end = platoon[-1]
require abs(relative heading of end from start) >= 60 deg