from gta import *

wiggle = (-10 deg, 10 deg)

ego = EgoCar with roadDeviation wiggle

c = Car visible, with roadDeviation resample(wiggle)

leftRight = Options([1.0, -1.0]) * (1.25, 2.75)

Car beyond c by leftRight @ (4, 10), \
	with roadDeviation resample(wiggle)