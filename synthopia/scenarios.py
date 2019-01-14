
import random
import math
import collections
import itertools
import time
import colorsys
from typing import NamedTuple
from enum import Enum

import numpy
import scipy.spatial
import PIL
import cv2

import center_detection
import img_modf
import messages

### Utilities

def argsToString(args):
	return '({})'.format(', '.join(str(a) for a in args))

### Distributions

def valueInContext(value, context):
	try:
		return value.evaluateIn(context)
	except AttributeError:
		return value

## Abstract distributions

class DefaultIdentityDict(dict):
	def __getitem__(self, key):
		if not isinstance(key, Samplable):		# to allow non-hashable objects
			return key
		return super().__getitem__(key)

	def __missing__(self, key):
		return key

class Samplable:
	def __init__(self, dependencies):
		deps = set()
		for dep in dependencies:
			if isinstance(dep, Samplable):
				deps.add(dep)
		self.dependencies = deps

	@staticmethod
	def sampleAll(quantities):
		subsamples = DefaultIdentityDict()
		for q in quantities:
			if q not in subsamples:
				subsamples[q] = q.sample(subsamples) if isinstance(q, Samplable) else q
		return { q: subsamples[q] for q in quantities }

	def sample(self, subsamples=None):
		if subsamples is None:
			subsamples = DefaultIdentityDict()
		for child in self.dependencies:
			if child not in subsamples:
				subsamples[child] = child.sample(subsamples)
		return self.sampleGiven(subsamples)

	def sampleGiven(self, value):
		return DefaultIdentityDict({ dep: value[dep] for dep in self.dependencies })

class Distribution(Samplable):
	"""Abstract class for distributions"""

	defaultValueType = float

	def __init__(self, *dependencies, valueType=None):
		props = set()
		for dep in dependencies:
			if hasattr(dep, 'requiredProperties'):
				props.update(dep.requiredProperties)
		super().__init__(dependencies)
		self.requiredProperties = props
		if valueType is None:
			valueType = self.defaultValueType
		self.valueType = valueType

	def clone(self):
		raise NotImplementedError('clone() not supported by this distribution')

	def evaluateIn(self, context):
		if self.requiredProperties:
			self.evaluateInner(context)
			self.requiredProperties = set()
		return self

	def evaluateInner(self, context):
		pass

	def __getattr__(self, name):
		return AttributeDistribution(name, self)

	def dependencyTree(self):
		l = [str(self)]
		for dep in self.dependencies:
			for line in dep.dependencyTree():
				l.append('  ' + line)
		return l

class ConstantDistribution(Distribution):
	"""Distribution concentrated on one value (delta distribution)"""
	def __init__(self, value):
		assert not isinstance(value, Distribution)
		super(ConstantDistribution, self).__init__()
		self.value = value

	def sampleGiven(self, value):
		return self.value

	def evaluateInner(self, context):
		self.value = valueInContext(self.value, context)

	def __getattr__(self, name):
		return getattr(self.value, name)

	def __str__(self):
		return str(self.value)

class CustomDistribution(Distribution):
	"""Distribution with a custom sampler given by an arbitrary function"""
	def __init__(self, sampler, *dependencies, name='CustomDistribution', evaluator=None):
		super(CustomDistribution, self).__init__(*dependencies)
		self.sampler = sampler
		self.name = name
		self.evaluator = evaluator

	def sampleGiven(self, value):
		return self.sampler(value)

	def evaluateInner(self, context):
		if self.evaluator is None:
			raise NotImplementedError('evaluateIn() not supported by this distribution')
		self.evaluator(self, context)

	def __str__(self):
		return '{}{}'.format(self.name, argsToString(self.dependencies))

class TupleDistribution(Distribution, collections.abc.Sequence):
	"""Distributions over tuples"""
	def __init__(self, *coordinates):
		super(TupleDistribution, self).__init__(*coordinates)
		self.coordinates = coordinates

	def __len__(self):
		return len(self.coordinates)

	def __getitem__(self, index):
		return self.coordinates[index]

	def sampleGiven(self, value):
		return tuple(value[coordinate] for coordinate in self.coordinates)

	def evaluateInner(self, context):
		self.coordinates = tuple(valueInContext(coord, context) for coord in self.coordinates)

	def __str__(self):
		return '({})'.format(', '.join(str(c) for c in self.coordinates))

def toDistribution(val, always=True):
	if isinstance(val, tuple):
		coords = [toDistribution(c, always=True) for c in val]
		needed = always or any(isinstance(c, Distribution) for c in coords)
		return TupleDistribution(*coords) if needed else val
	return val

class FunctionDistribution(Distribution):
	"""Distribution resulting from passing distributions to a function"""
	def __init__(self, func, args):
		args = tuple(toDistribution(arg) for arg in args)
		super(FunctionDistribution, self).__init__(*args)
		self.function = func
		self.arguments = args

	def sampleGiven(self, value):
		args = tuple(value[arg] for arg in self.arguments)
		return self.function(*args)

	def evaluateInner(self, context):
		self.function = valueInContext(self.function, context)
		self.arguments = tuple(valueInContext(arg, context) for arg in self.arguments)

	def __str__(self):
		return '{}{}'.format(self.function.__name__, argsToString(self.arguments))

def distributionFunction(method):
	def helper(*args):
		args = tuple(toDistribution(arg, always=False) for arg in args)
		if any(isinstance(arg, Distribution) for arg in args):
			return FunctionDistribution(method, args)
		else:
			return method(*args)
	return helper

class MethodDistribution(Distribution):
	"""Distribution resulting from passing distributions to a method of a fixed object"""
	def __init__(self, method, obj, args):
		args = tuple(toDistribution(arg) for arg in args)
		super(MethodDistribution, self).__init__(*args)
		self.method = method
		self.object = obj
		self.arguments = args

	def sampleGiven(self, value):
		args = [value[arg] for arg in self.arguments]
		return self.method(self.object, *args)

	def evaluateInner(self, context):
		self.object = valueInContext(self.object, context)
		self.arguments = tuple(valueInContext(arg, context) for arg in self.arguments)

	def __str__(self):
		return '{}.{}{}'.format(self.object, self.method.__name__, argsToString(self.arguments))

def distributionMethod(method):
	def helper(self, *args):
		args = tuple(toDistribution(arg, always=False) for arg in args)
		if any(isinstance(arg, Distribution) for arg in args):
			return MethodDistribution(method, self, args)
		else:
			return method(self, *args)
	return helper

class AttributeDistribution(Distribution):
	"""Distribution resulting from accessing an attribute of a distribution"""
	def __init__(self, attribute, obj):
		super(AttributeDistribution, self).__init__(obj)
		self.attribute = attribute
		self.object = obj

	def sampleGiven(self, value):
		obj = value[self.object]
		return getattr(obj, self.attribute)

	def evaluateInner(self, context):
		self.object = valueInContext(self.object, context)

	def __str__(self):
		return '{}.{}'.format(self.object, self.attribute)

class OperatorDistribution(Distribution):
	"""Distribution resulting from applying an operator to one or more distributions"""
	def __init__(self, operator, obj, operands):
		operands = tuple(toDistribution(arg) for arg in operands)
		super(OperatorDistribution, self).__init__(obj, *operands)
		self.operator = operator
		self.object = obj
		self.operands = operands

	def sampleGiven(self, value):
		first = value[self.object]
		rest = (value[child] for child in self.operands)
		op = getattr(first, self.operator)
		return op(*rest)

	def evaluateInner(self, context):
		self.object = valueInContext(self.object, context)
		self.operands = tuple(valueInContext(arg, context) for arg in self.operands)

	def __str__(self):
		return '{}.{}{}'.format(self.object, self.operator, argsToString(self.operands))

allowedOperators = [
	'__neg__',
	'__pos__',
	'__abs__',
	'__lt__', '__le__',
	'__eq__', '__ne__',
	'__gt__', '__ge__',
	'__add__', '__radd__',
	'__sub__', '__rsub__',
	'__mul__', '__rmul__',
	'__truediv__', '__rtruediv__',
	'__floordiv__', '__rfloordiv__',
	'__mod__', '__rmod__',
	'__divmod__', '__rdivmod__',
	'__pow__', '__rpow__',
	'__round__',
	'__len__',
	'__getitem__'
	]
def makeOperatorHandler(op):
	def handler(self, *args):
		return OperatorDistribution(op, self, args)
	return handler
for op in allowedOperators:
	setattr(Distribution, op, makeOperatorHandler(op))

## Functions that can accept distributions

@distributionFunction
def sin(x):
	return math.sin(x)

@distributionFunction
def cos(x):
	return math.cos(x)

@distributionFunction
def hypot(x, y):
	return math.hypot(x, y)

@distributionFunction
def max(*args):
	return __builtins__['max'](*args)

@distributionFunction
def min(*args):
	return __builtins__['min'](*args)

## Simple distributions

class Range(Distribution):
	"""Uniform distribution over a range"""
	def __init__(self, low, high):
		super(Range, self).__init__(low, high)
		self.low = low
		self.high = high

	def __contains__(self, obj):
		return low <= obj and obj <= high

	def clone(self):
		return type(self)(self.low, self.high)

	def sampleGiven(self, value):
		return random.uniform(value[self.low], value[self.high])

	def evaluateInner(self, context):
		self.low = valueInContext(self.low, context)
		self.high = valueInContext(self.high, context)

	def __str__(self):
		return 'Range({}, {})'.format(self.low, self.high)

class Normal(Distribution):
	"""Normal distribution"""
	def __init__(self, mean, stddev):
		super(Normal, self).__init__(mean, stddev)
		self.mean = mean
		self.stddev = stddev

	def clone(self):
		return type(self)(self.mean, self.stddev)

	def sampleGiven(self, value):
		return random.gauss(value[self.mean], value[self.stddev])

	def evaluateInner(self, context):
		self.mean = valueInContext(self.mean, context)
		self.stddev = valueInContext(self.stddev, context)

	def __str__(self):
		return 'Normal({}, {})'.format(self.mean, self.stddev)

class Options(Distribution):
	"""Distribution over a finite list of options.
	Specified by a dict giving probabilities; otherwise uniform over a given iterable."""
	def __init__(self, opts):
		if isinstance(opts, dict):
			self.options = []
			self.weights = dict()
			ordered = []
			for opt, prob in opts.items():
				opt = toDistribution(opt)
				self.options.append(opt)
				self.weights[opt] = prob
				ordered.append(prob)
			self.cumulativeWeights = tuple(itertools.accumulate(ordered))
		else:
			self.options = tuple(toDistribution(opt) for opt in opts)
			self.cumulativeWeights = None
		super(Options, self).__init__(*self.options)

	def __getattr__(self, name):
		return AttributeDistribution(name, self)

	def clone(self):
		return type(self)(self.weights if self.cumulativeWeights is not None else self.options)

	def sampleGiven(self, value):
		opts = [value[opt] for opt in self.options]
		return random.choices(opts, cum_weights=self.cumulativeWeights)[0]

	def evaluateInner(self, context):
		self.options = [valueInContext(opt, context) for opt in self.options]

	def __str__(self):
		if self.cumulativeWeights is not None:
			parts = ('{}: {}'.format(str(opt), self.weights[opt]) for opt in self.options)
			return 'Options({{{}}})'.format(', '.join(parts))
		else:
			return 'Options{}'.format(argsToString(self.options))

### Simulator interface

class Simulator(object):
	"""docstring for Simulator"""

	@staticmethod
	def Vehicle(model, color, location, heading):
		loc3 = Simulator.langToSimCoords(location)
		heading = Simulator.langToSimHeading(heading)
		scol = list(CarColor.realToByte(color))
		return messages.Vehicle(model.name, scol, loc3, heading)

	@staticmethod
	def Config(cameraLocation, cameraHeading, time, weather, vehicles):
		loc3 = Simulator.langToSimCoords(cameraLocation)
		cameraHeading = Simulator.langToSimHeading(cameraHeading)
		time = int(round(time))
		minute = time % 60
		hour = int((time - minute) / 60)
		assert hour < 24
		return messages.Formal_Config(loc3, [hour, minute], weather, vehicles, cameraHeading)
	
	@staticmethod
	def langToSimCoords(point):
		x, y = point
		return [x, y, 60]

	@staticmethod
	def langToSimHeading(heading):
		h = math.degrees(heading)
		return (h + 360) % 360

### Map and cars

def addVectors(a, b):
	ax, ay = a[0], a[1]
	bx, by = b[0], b[1]
	return (ax + bx, ay + by)

def averageVectors(a, b, weight=0.5):
	ax, ay = a[0], a[1]
	bx, by = b[0], b[1]
	aw, bw = 1.0 - weight, weight
	return (ax * aw + bx * bw, ay * aw + by * bw)

def rotateVector(vector, angle):
	x, y = vector
	c, s = cos(angle), sin(angle)
	return ((c * x) - (s * y), (s * x) + (c * y))

def findMinMax(iterable):
	minv = float('inf')
	maxv = float('-inf')
	for val in iterable:
		if val < minv:
			minv = val
		if val > maxv:
			maxv = val
	return (minv, maxv)

def radialToCartesian(point, radius, heading):
	angle = heading + (math.pi / 2.0)
	rx, ry = radius * cos(angle), radius * sin(angle)
	return (point[0] + rx, point[1] + ry)

def positionRelativeToPoint(point, heading, offset):
	ro = rotateVector(offset, heading)
	return addVectors(point, ro)

def viewAngleToPoint(point, base, heading):
	x, y = base
	ox, oy = point
	a = math.atan2(oy - y, ox - x) - (heading + (math.pi / 2.0))
	if a < -math.pi:
		a += math.tau
	elif a > math.pi:
		a -= math.tau
	assert -math.pi <= a and a <= math.pi
	return a

def apparentHeadingAtPoint(point, heading, base):
	x, y = base
	ox, oy = point
	a = (heading + (math.pi / 2.0)) - math.atan2(oy - y, ox - x)
	if a < -math.pi:
		a += math.tau
	elif a > math.pi:
		a -= math.tau
	assert -math.pi <= a and a <= math.pi
	return a

def circumcircleOfAnnulus(center, heading, angle, minDist, maxDist):
	m = (minDist + maxDist) / 2.0
	g = (maxDist - minDist) / 2.0
	h = m * math.sin(angle / 2.0)
	h2 = h * h
	d = math.sqrt(h2 + (m * m))
	r = math.sqrt(h2 + (g * g))
	return radialToCartesian(center, d, heading), r

def pointIsInCone(point, base, heading, angle):
	va = viewAngleToPoint(point, base, heading)
	return (abs(va) <= angle / 2.0)

class Map(object):
	"""Represents roads and obstacles"""
	def __init__(self, imagePath, Ax, Ay, Bx, By):
		super(Map, self).__init__()
		self.Ax, self.Ay = Ax, Ay
		self.Bx, self.By = Bx, By
		if imagePath != None:
			startTime = time.time()
			# open image
			image = PIL.Image.open(imagePath)
			self.sizeX, self.sizeY = image.size
			# create version of image for display
			de = img_modf.get_edges(image).convert('RGB')
			self.displayImage = cv2.cvtColor(numpy.array(de), cv2.COLOR_RGB2BGR)
			# detect edges of roads
			ed = center_detection.compute_midpoints(img_data=image, kernelsize=5)
			self.edgeData = { self.mapToLangCoords((x, y)): datum for (y, x), datum in ed.items() }
			self.orderedCurbPoints = list(self.edgeData.keys())
			# build k-D tree
			self.edgeTree = scipy.spatial.cKDTree(self.orderedCurbPoints)
			# identify points on roads
			self.roadArray = numpy.array(img_modf.convert_black_white(img_data=image).convert('L'), dtype=int)
			roadY, roadX = numpy.where(self.roadArray == 0)
			self.orderedRoadPoints = [self.mapToLangCoords(point) for point in zip(roadX, roadY)]
			print('created map from image in {:.2f} seconds'.format(time.time() - startTime))

	@staticmethod
	def fromFile(path):
		startTime = time.time()
		with numpy.load(path) as data:
			Ax, Ay, Bx, By, sizeX, sizeY = data['misc']
			m = Map(None, Ax, Ay, Bx, By)
			m.sizeX, m.sizeY = sizeX, sizeY
			m.displayImage = data['displayImage']
			
			m.edgeData = { tuple(e): center_detection.EdgeData(*rest) for e, *rest in data['edges'] }
			m.orderedCurbPoints = list(m.edgeData.keys())
			m.edgeTree = scipy.spatial.cKDTree(m.orderedCurbPoints)		# rebuild k-D tree

			m.roadArray = data['roadArray']
			roadY, roadX = numpy.where(m.roadArray == 0)
			m.orderedRoadPoints = [m.mapToLangCoords(point) for point in zip(roadX, roadY)]
			print('loaded map in {:.2f} seconds'.format(time.time() - startTime))
			return m

	def dumpToFile(self, path):
		misc = numpy.array((self.Ax, self.Ay, self.Bx, self.By, self.sizeX, self.sizeY))
		edges = numpy.array([(edge,) + tuple(datum) for edge, datum in self.edgeData.items()])
		roadArray = self.roadArray

		numpy.savez_compressed(path,
			misc=misc, displayImage=self.displayImage,
			edges=edges, roadArray=self.roadArray)

	def mapToLangCoords(self, point):
		x, y = point[0], point[1]
		return ((self.Ax * x) + self.Bx, (self.Ay * y) + self.By)

	def mapToLangHeading(self, heading):
		return heading - (math.pi / 2)

	def langToMapCoords(self, point):
		x, y = point[0], point[1]
		return ((x - self.Bx) / self.Ax, (y - self.By) / self.Ay)

	def langToMapHeading(self, heading):
		return heading + (math.pi / 2)

	def inBounds(self, point):
		x, y = self.langToMapCoords(point)
		nx = round(x)
		if nx < 0 or nx >= self.sizeX:
			return False
		ny = round(y)
		if ny < 0 or ny >= self.sizeY:
			return False
		return True

	def nearestGridPoint(self, point):
		x, y = self.langToMapCoords(point)
		nx = int(round(x))
		if nx < 0 or nx >= self.sizeX:
			return None
		ny = int(round(y))
		if ny < 0 or ny >= self.sizeY:
			return None
		return (nx, ny)

	@distributionMethod
	def roadHeadingAt(self, point):
		# find closest edge
		distance, location = self.edgeTree.query(point)
		closest = tuple(self.edgeTree.data[location])
		# get direction of edge
		return self.mapToLangHeading(self.edgeData[closest].tangent)

	@distributionFunction
	def alongRoadFrom(self, point, distance, steps=4):
		# TODO use something better than Euler's method?
		step = distance / steps
		for i in range(steps):
			point = radialToCartesian(point, step, self.roadHeadingAt(point))
		return point

	@distributionFunction
	def positionRelativeToRoad(self, point, offset):
		return positionRelativeToPoint(point, self.roadHeadingAt(point), offset)

	def pointIsOnRoad(self, point):
		x, y = point
		return (self.roadArray[y, x] == 0)

	def uniformPointOnRoad(self):
		return CustomDistribution(lambda values: random.choice(self.orderedRoadPoints),
			name='uniformPointOnRoad')

	def uniformPointOnCurb(self):
		return CustomDistribution(lambda values: random.choice(self.orderedCurbPoints),
			name='uniformPointOnCurb')

	def uniformCurbVisibleFrom(self, car, minDist=5, maxDist=50):
		# TODO improve this procedure?
		def sampler(value):
			c = value(car)
			center, radius = circumcircleOfAnnulus(c.position, c.heading, c.viewAngle, minDist, maxDist)
			possibles = (tuple(self.edgeTree.data[i]) for i in self.edgeTree.query_ball_point(center, radius))
			curbs = [p for p in possibles if pointIsInCone(p, c.position, c.heading, c.viewAngle)]
			if len(curbs) == 0:
				return (-1000, -1000)		# TODO more principled way to ensure rejection?
			return random.choice(curbs)
		return CustomDistribution(sampler, car, name='uniformCurbVisibleFrom')

	def rectIsOnRoad(self, rect):
		# TODO improve this procedure!
		# Fast check
		for c in rect.corners:
			gp = self.nearestGridPoint(c)
			if gp is None or not self.pointIsOnRoad(gp):
				return False
		# Slow check
		x, y = self.nearestGridPoint(rect.corners[0])
		minx = maxx = x
		miny = maxy = y
		for c in rect.corners[1:]:
			x, y = self.nearestGridPoint(c)
			if x < minx:
				minx = x
			if x > maxx:
				maxx = x
			if y < miny:
				miny = y
			if y > maxy:
				maxy = y
		for x in range(minx, maxx+1):
			for y in range(miny, maxy+1):
				p = (x, y)
				if not self.pointIsOnRoad(p) and rect.containsPoint(p):
					return False
		return True

	def show(self, plt):
		plt.imshow(self.displayImage)

	def zoomAround(self, plt, rects, fudge=2, minSize=0):
		positions = (self.langToMapCoords(r.position) for r in rects)
		x, y = zip(*positions)
		minx, maxx = findMinMax(x)
		miny, maxy = findMinMax(y)
		sx = fudge * (maxx - minx)
		sy = fudge * (maxy - miny)
		ms = minSize / min(abs(self.Ax), abs(self.Ay))
		s = max(sx, sy, ms) / 2.0
		cx = (maxx + minx) / 2.0
		cy = (maxy + miny) / 2.0
		plt.xlim(cx - s, cx + s)
		plt.ylim(cy + s, cy - s)

class Rectangle:
	"""A rectangular region (not necessarily axis-aligned)"""
	def __init__(self, position, heading, width, height):
		self.position = position
		self.heading = heading
		self.width = width
		self.height = height
		hw = width / 2.0
		hh = height / 2.0
		self.hw = hw
		self.hh = hh
		self.radius = hypot(hw, hh)
		self.corners = tuple(addVectors(self.position, rotateVector(v, self.heading))
			for v in ((hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)))

	def containsPoint(self, point):
		x, y = rotateVector(numpy.array(point) - self.position, -self.heading)
		return abs(x) <= self.hw and abs(y) <= self.hh

	def intersects(self, rect):
		# Quick check by bounding circles
		x, y = self.position
		rx, ry = rect.position
		dx, dy = rx - x, ry - y
		rr = self.radius + rect.radius
		if (dx * dx) + (dy * dy) > (rr * rr):
			return False
		# Check for separating line parallel to our edges
		if self.edgeSeparates(self, rect):
			return False
		# Check for separating line parallel to rect's edges
		if self.edgeSeparates(rect, self):
			return False
		return True

	@staticmethod
	def edgeSeparates(rectA, rectB):
		"""Whether an edge of rectA separates it from rectB"""
		rc = [rotateVector(numpy.array(c) - rectA.position, -rectA.heading) for c in rectB.corners]
		x, y = zip(*rc)
		minx, maxx = findMinMax(x)
		miny, maxy = findMinMax(y)
		if maxx < -rectA.hw or rectA.hw < minx:
			return True
		if maxy < -rectA.hh or rectA.hh < miny:
			return True
		return False

	def __str__(self):
		return 'Rectangle({},{},{},{})'.format(self.position, self.heading, self.width, self.height)

class CarModel(object):
	def __init__(self, name, width, height, viewAngle=math.radians(90)):
		super(CarModel, self).__init__()
		self.name = name
		self.width = width
		self.height = height
		self.viewAngle = viewAngle

	@classmethod
	def uniformModel(self):
		return Options(self.modelProbs.keys())

	@classmethod
	def egoModel(self):
		return self.models['BLISTA']

	@classmethod
	def defaultModel(self):
		return Options(self.modelProbs)

	def __str__(self):
		return self.name

CarModel.modelProbs = {
	CarModel('BLISTA', 1.75871, 4.10139): 1,
	CarModel('BUS', 2.9007, 13.202): 0,
	CarModel('NINEF', 2.07699, 4.50658): 1,
	CarModel('ASEA', 1.83066, 4.45861): 1,
	CarModel('BALLER', 2.10791, 5.10333): 1,
	CarModel('BISON', 2.29372, 5.4827): 1,
	CarModel('BUFFALO', 2.04265, 5.07782): 1,
	CarModel('BOBCATXL', 2.37944, 5.78222): 1,
	CarModel('DOMINATOR', 1.9353, 4.9355): 1,
	CarModel('GRANGER', 3.02698, 5.94577): 1,
	CarModel('JACKAL', 2.00041, 4.91436): 1,
	CarModel('ORACLE', 2.07787, 5.12544): 1,
	CarModel('PATRIOT', 2.26679, 5.13695): 1,
	CarModel('PRANGER', 3.02698, 5.94577): 1
}
CarModel.models = { model.name: model for model in CarModel.modelProbs }

class CarColor:
	@staticmethod
	def rgb(r, g, b):
		return (r, g, b)

	@staticmethod
	def byteToReal(color):
		return tuple(c / 255.0 for c in color)

	@staticmethod
	def realToByte(color):
		return tuple(int(round(255 * c)) for c in color)

	@staticmethod
	def uniformColor():
		return TupleDistribution(Range(0, 1), Range(0, 1), Range(0, 1))

	@staticmethod
	def defaultColor():
		"""Base color distribution estimated from 2012 DuPont survey archived at:
		https://web.archive.org/web/20121229065631/http://www2.dupont.com/Media_Center/en_US/color_popularity/Images_2012/DuPont2012ColorPopularity.pdf"""
		baseColors = {
			(248, 248, 248): 0.24,	# white
			(50, 50, 50): 0.19,		# black
			(188, 185, 183): 0.16,	# silver
			(130, 130, 130): 0.15,	# gray
			(194, 92, 85): 0.10,	# red
			(75, 119, 157): 0.07,	# blue
			(197, 166, 134): 0.05,	# brown/beige
			(219, 191, 105): 0.02,	# yellow/gold
			(68, 160, 135): 0.02,	# green
		}
		converted = { CarColor.byteToReal(color): prob for color, prob in baseColors.items() }
		baseColor = Options(converted)
		# TODO improve this?
		hueNoise = Normal(0, 0.1)
		satNoise = Normal(0, 0.1)
		lightNoise = Normal(0, 0.1)
		return NoisyColorDistribution(baseColor, hueNoise, satNoise, lightNoise)

class NoisyColorDistribution(Distribution):
	def __init__(self, baseColor, hueNoise, satNoise, lightNoise):
		super(NoisyColorDistribution, self).__init__(baseColor, hueNoise, satNoise, lightNoise)
		self.baseColor = baseColor
		self.hueNoise = hueNoise
		self.satNoise = satNoise
		self.lightNoise = lightNoise

	@staticmethod
	def addNoiseTo(color, hueNoise, lightNoise, satNoise):
		hue, lightness, saturation = colorsys.rgb_to_hls(*color)
		hue = max(0, min(1, hue + hueNoise))
		lightness = max(0, min(1, lightness + lightNoise))
		saturation = max(0, min(1, saturation + satNoise))
		return colorsys.hls_to_rgb(hue, lightness, saturation)

	def sampleGiven(self, value):
		bc = value[self.baseColor]
		return self.addNoiseTo(bc, value[self.hueNoise],
			value[self.lightNoise], value[self.satNoise])

	def evaluateInner(self, context):
		self.baseColor = valueInContext(self.baseColor, context)
		self.hueNoise = valueInContext(self.hueNoise, context)
		self.satNoise = valueInContext(self.satNoise, context)
		self.lightNoise = valueInContext(self.lightNoise, context)

class Car(Rectangle, Distribution):
	"""Represents a car"""
	def __init__(self, position, heading, model=None, color=None):
		position = toDistribution(position, always=False)
		if model is None:
			model = CarModel.defaultModel()
		if color is None:
			color = CarColor.defaultColor()
		super(Car, self).__init__(position, heading, model.width, model.height)
		Distribution.__init__(self, position, heading, model, color)
		self.model = model
		self.color = color
		self.viewAngle = model.viewAngle

	def sampleGiven(self, value):
		return Car(value[self.position], value[self.heading],
			model=value[self.model], color=value[self.color])

	def relativePosition(self, offset):
		return positionRelativeToPoint(self.position, self.heading, offset)

	def relativeHeading(self, heading):
		a = heading - self.heading
		if a > math.pi:
			a -= math.tau
		elif a < -math.pi:
			a += math.tau
		assert -math.pi <= a and a <= math.pi
		return a

	def relativeFrontPosition(self, offset):
		toFront = (0, self.hh)
		return positionRelativeToPoint(self.position, self.heading, addVectors(toFront, offset))

	def distanceTo(self, other):
		x, y = self.position
		ox, oy = other.position
		return math.hypot(ox - x, oy - y)

	def viewAngleTo(self, other):
		return viewAngleToPoint(other.position, self.position, self.heading)

	def apparentHeadingOf(self, other):
		return apparentHeadingAtPoint(other.position, other.heading, self.position)

	def uniformVisiblePoint(self, minDist=5, maxDist=50):
		return UniformAnnulusDistribution(self.position, self.heading, self.viewAngle, minDist, maxDist)

	def canSee(self, other):
		for corner in other.corners:
			if pointIsInCone(corner, self.position, self.heading, self.viewAngle):
				return True
		return False

	def isOncoming(self, other, threshold=math.radians(60)):
		return pointIsInCone(self.position, other.position, other.heading, threshold)

	def isConcrete(self):
		return len(self.dependencies) == 0

	def toSimulatorVehicle(self):
		if not self.isConcrete():
			raise RuntimeError('called toSimulatorVehicle() on symbolic Car')
		return Simulator.Vehicle(self.model, self.color, self.position, self.heading)

	def show(self, map, plt, highlight=False):
		if not self.isConcrete():
			raise RuntimeError('tried to show() symbolic Car')
		mypos = map.langToMapCoords(self.position)

		if highlight:
			# Circle around car
			rad = 1.5 * max(self.width, self.height)
			c = plt.Circle(mypos, rad, color='g', fill=False)
			plt.gca().add_artist(c)
			# View cone
			ha = self.viewAngle / 2.0
			for angle in (-ha, ha):
				p = radialToCartesian(self.position, 20, self.heading + angle)
				edge = [mypos, map.langToMapCoords(p)]
				x, y = zip(*edge)
				plt.plot(x, y, 'b:')

		corners = [map.langToMapCoords(corner) for corner in self.corners]
		x, y = zip(*corners)
		plt.fill(x, y, color=self.color)
		plt.plot(x + (x[0],), y + (y[0],), color="w", linewidth=1)

		frontMid = averageVectors(corners[0], corners[1])
		baseTriangle = [frontMid, corners[2], corners[3]]
		triangle = [averageVectors(p, mypos, weight=0.5) for p in baseTriangle]
		x, y = zip(*triangle)
		plt.fill(x, y, "w")
		plt.plot(x + (x[0],), y + (y[0],), color="k", linewidth=1)

	def __str__(self):
		return 'Car({}, {}, {}, {})'.format(self.position, self.heading, self.model, self.color)

### Scenarios

class Scene:
	def __init__(self, m, cars, egoCar, time, weather):
		self.map = m
		self.cars = cars
		self.egoCar = egoCar
		self.time = time
		self.weather = weather

	def show(self, zoom=None, minSize=40):
		import matplotlib.pyplot as plt
		# display map
		self.map.show(plt)
		# draw cars
		for car in self.cars:
			car.show(self.map, plt, highlight=(car is self.egoCar))
		# zoom in if requested
		if zoom != None:
			self.map.zoomAround(plt, self.cars, fudge=zoom, minSize=minSize)
		plt.show()

	def toSimulatorConfig(self):
		vehicles = [car.toSimulatorVehicle() for car in self.cars if car is not self.egoCar]
		cameraLoc = self.egoCar.position
		cameraHeading = self.egoCar.heading
		return Simulator.Config(cameraLoc, cameraHeading, self.time, self.weather, vehicles)

	def __str__(self):
		return 'Scene({},{},{},{})'.format(self.cars, self.egoCar, self.time, self.weather)

class RejectionException(Exception):
	pass

class Scenario(Samplable):
	"""A scenario consisting of a map, cars, and global state"""
	def __init__(self, m, cars, egoCar, time=None, weather=None,
		requirements=None, requirementDeps=None, requireVisible=True):
		if time is None:
			time = self.defaultTime()
		if weather is None:
			weather = self.defaultWeather()
		self.cars = list(cars)
		self.requirements = [] if requirements is None else requirements
		self.requirementDeps = [] if requirementDeps is None else requirementDeps
		super().__init__(set(self.cars) | set(self.requirementDeps) | {time, weather})
		self.map = m
		self.egoCar = egoCar
		if egoCar not in self.cars:
			raise RuntimeError('ego car must be part of scenario!')
		self.time = time
		self.weather = weather
		self.requireVisible = requireVisible

	@staticmethod
	def defaultTime():
		return Range(0 * 60, 24 * 60)	# 0000 to 2400 hours

	@staticmethod
	def defaultWeather():
		return Options({
			'NEUTRAL': 5,
			'CLEAR': 15,
			'EXTRASUNNY': 20,
			'CLOUDS': 15,
			'OVERCAST': 15,
			'RAIN': 5,
			'THUNDER': 5,
			'CLEARING': 5,
			'FOGGY': 5,
			'SMOG': 5,
			'XMAS': 1.25,
			'SNOWLIGHT': 1.25,
			'BLIZZARD': 1.25,
			'SNOW': 1.25
			})

	def addRequirement(self, r):
		self.requirements.append(r)

	def generate(self, maxIterations=2000):
		startTime = time.time()
		cars = self.cars
		reject = True
		iterations = 0
		activeReqs = [req for req, prob in self.requirements if random.random() <= prob]
		while reject:
			if iterations >= maxIterations:
				raise RuntimeError('failed to generate scenario in {} iterations'.format(iterations))
			iterations += 1
			try:
				sample = self.sample()
			except RejectionException:
				reject = True
				continue
			reject = False
			ego = sample[self.egoCar]
			# Check built-in requirements
			for i in range(len(cars)):
				vi = sample[cars[i]]
				# Require car to be on road
				if not self.map.rectIsOnRoad(vi):
					reject = True
					break
				# Require car to be visible from ego car
				if self.requireVisible and vi is not ego and not ego.canSee(vi):
					reject = True
					break
				# Require car to not intersect another car
				for j in range(i):
					vj = sample[cars[j]]
					if vi.intersects(vj):
						reject = True
						break
				if reject:
					break
			if reject:
				continue
			# Check user-specified requirements
			for req in activeReqs:
				if not req(sample):
					reject = True
					break
		print('generated scene in {} iterations, {:.4g} seconds'.format(iterations, time.time() - startTime))

		return Scene(self.map, [sample[car] for car in cars], ego,
			sample[self.time], sample[self.weather])


