import random
from scenarios import Map, CarModel, CarColor, NoisyColorDistribution, Simulator

# Load map and set up regions, etc.
mapFilename = 'map.npz'
try:
	m = Map.fromFile(mapFilename)
except FileNotFoundError:
	print('Saved map not found; generating from image... (may take several minutes)')
	m = Map('pics/gta_map.png',
		Ax=1.515151515151500 / 2, Ay=-1.516919486581100 / 2,
		Bx=-700, By=500)
	m.dumpToFile(mapFilename)
	print('map saved to "{}"'.format(mapFilename))

roadDirection = VectorField('roadDirection', m.roadHeadingAt)
road = PointSetRegion('road', m.orderedRoadPoints, orientation=roadDirection)
curb = PointSetRegion('curb', m.orderedCurbPoints, orientation=roadDirection)

# Define custom type of object for cars
constructor Car:
	position: Point on road
	heading: (roadDirection at self.position) + self.roadDeviation
	roadDeviation: 0
	width: self.model.width
	height: self.model.height
	viewAngle: 80 deg
	visibleDistance: 30
	model: CarModel.defaultModel()
	color: CarColor.defaultColor()

	mutator[additive]: CarColorMutator()

	def toSimulatorVehicle(self):
		return Simulator.Vehicle(self.model, self.color, self.position, self.heading)

class CarColorMutator(Mutator):
	def appliedTo(self, obj):
		hueNoise = random.gauss(0, 0.05)
		satNoise = random.gauss(0, 0.05)
		lightNoise = random.gauss(0, 0.05)
		color = NoisyColorDistribution.addNoiseTo(obj.color, hueNoise, lightNoise, satNoise)
		return tuple([obj.copyWith(color=color), True])		# allow further mutation

# Convenience subclass with defaults for ego cars
constructor EgoCar(Car):
	model: CarModel.egoModel()

# Convenience subclass for buses
constructor Bus(Car):
	model: CarModel.models['BUS']

# Convenience subclass for compact cars
constructor Compact(Car):
	model: CarModel.models['BLISTA']

# Helper function for making platoons
def createPlatoonAt(car, numCars, model=None, dist=(2, 8), shift=(-0.5, 0.5), wiggle=0):
	cars = [car]
	lastCar = car
	for i in range(numCars-1):
		center = follow roadDirection from (front of lastCar) for resample(dist)
		pos = OrientedPoint at (center offset by shift @ 0), facing resample(wiggle) relative to roadDirection
		lastCar = Car ahead of pos, with model (car.model if model is None else resample(model))
		cars.append(lastCar)
	return cars
