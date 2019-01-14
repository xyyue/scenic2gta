
from scenarios import *

# Load map

try:
	m = Map.fromFile('map.npz')
except FileNotFoundError:
	m = Map('pics/gta_map.png',
		Ax=0.758725341426, Ay=-0.759878419452888,
		Bx=-1038.694992412747, By=79.787234042553209)
	m.dumpToFile('map.npz')

# Set up scenario

pos = m.uniformPointOnRoad()
h = m.roadHeadingAt(pos)
c1 = Car(pos, h)

p2 = c1.relativePosition((0, 30))
h2 = m.roadHeadingAt(p2)
c2 = Car(p2, h2)

offset = Range(-20, 20)
p3 = c2.relativePosition((offset, 0))
h3 = m.roadHeadingAt(p3)
c3 = Car(p3, h3)

s = Scenario(m, [c1, c2, c3], egoCar=c1)

# Generate concrete configurations

# v = s.generate()
# config = v.toSimulatorConfig()

while True:
	v = s.generate()
	v.show(zoom=3)
