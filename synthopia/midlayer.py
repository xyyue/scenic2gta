
import math
import inspect
import collections
import random
import scipy.spatial

from scenarios import Distribution, Range, CustomDistribution, makeOperatorHandler, argsToString
from scenarios import distributionMethod, sin, cos, hypot, findMinMax, RejectionException
from scenarios import apparentHeadingAtPoint, positionRelativeToPoint, pointIsInCone, averageVectors
from scenarios import circumcircleOfAnnulus

from scenarios import valueInContext, Samplable		# TODO move these here!!!
from scenarios import max, min

### Internal stuff

class RuntimeParseError(Exception):
	pass

class DelayedArgument(Distribution):
	def __init__(self, deps, value):
		super().__init__(value)
		self.value = value
		self.requiredProperties = deps
		self.evaluated = False

	def copy(self):
		return DelayedArgument(self.requiredProperties, self.value)

	def evaluateIn(self, context):
		if not self.evaluated:
			assert all(hasattr(context, dep) for dep in self.requiredProperties)
			self.value = valueInContext(self.value(context), context)
			self.evaluated = True
			self.requiredProperties = set()
		return self.value

	def __getattr__(self, name):
		return DelayedArgument(self.requiredProperties,
			lambda context: getattr(self.evaluateIn(context), name))

	def __call__(self, *args):
		dargs = [toDelayedArgument(arg) for arg in args]
		return DelayedArgument(self.requiredProperties.union(*(darg.requiredProperties for darg in dargs)),
			lambda context: self.evaluateIn(context)(*(darg.evaluateIn(context) for darg in dargs)))

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
def makeDelayedOperatorHandler(op):
	def handler(self, *args):
		dargs = [toDelayedArgument(arg) for arg in args]
		return DelayedArgument(self.requiredProperties.union(*(darg.requiredProperties for darg in dargs)),
			lambda context: getattr(self.evaluateIn(context), op)(*(darg.evaluateIn(context) for darg in dargs))) 
	return handler
for op in allowedOperators:
	setattr(DelayedArgument, op, makeDelayedOperatorHandler(op))

class TypeChecker(DelayedArgument):
	def __init__(self, arg, types, error):
		def check(context):
			val = arg.evaluateIn(context)
			return coerceToAny(val, types, error)
		super().__init__(arg.requiredProperties, check)
		self.inner = arg
		self.types = types

	def __str__(self):
		return 'TypeChecker({},{})'.format(self.inner, self.types)

class TypeEqualityChecker(DelayedArgument):
	def __init__(self, arg, checkA, checkB, error):
		arg = toDelayedArgument(arg)
		assert requiredProperties(checkA) <= arg.requiredProperties
		assert requiredProperties(checkB) <= arg.requiredProperties
		def check(context):
			ca = valueInContext(checkA, context)
			cb = valueInContext(checkB, context)
			assert not requiredProperties(ca) and not requiredProperties(cb)
			if underlyingType(ca) is not underlyingType(cb):
				raise RuntimeParseError(error)
			return arg.evaluateIn(context)
		super().__init__(arg.requiredProperties, check)
		self.inner = arg
		self.checkA = checkA
		self.checkB = checkB

	def __str__(self):
		return 'TypeEqualityChecker({},{},{})'.format(self.inner, self.checkA, self.checkB)

def requiredProperties(thing):
	if hasattr(thing, 'requiredProperties'):
		return thing.requiredProperties
	return set()

def dependencies(thing):
	if hasattr(thing, 'dependencies'):
		return thing.dependencies
	return []

def toDelayedArgument(thing):
	if isinstance(thing, DelayedArgument):
		return thing
	return DelayedArgument(set(), lambda context: thing)

class Specifier(object):
	"""docstring for Specifier"""
	def __init__(self, prop, value, deps=None, optionals={}):
		super().__init__()
		self.property = prop
		self.value = toDelayedArgument(value).copy()	# TODO improve?
		if deps is None:
			deps = set()
		deps |= requiredProperties(value)
		assert prop not in deps
		self.requiredProperties = deps
		self.optionals = optionals

	def applyTo(self, obj, optionals):
		val = self.value.evaluateIn(obj)
		assert not isinstance(val, DelayedArgument)
		setattr(obj, self.property, val)
		for opt in optionals:
			assert opt in self.optionals
			setattr(obj, opt, getattr(val, opt))

def normalizeAngle(angle):
	while angle > math.pi:
		angle -= math.tau
	while angle < -math.pi:
		angle += math.tau
	assert -math.pi <= angle and angle <= math.pi
	return angle

#### Support for language constructs

### Various non-constructable types

# class Range(object):
# 	def __init__(self, low, high):
# 		super().__init__()
# 		self.low = low
# 		self.high = high

class VectorDistribution(Distribution):
	defaultValueType = None		# will be set after Vector is defined

	def toVector(self):
		return self

class CustomVectorDistribution(VectorDistribution):
	"""Distribution with a custom sampler given by an arbitrary function"""
	def __init__(self, sampler, *dependencies, name='CustomVectorDistribution', evaluator=None):
		super().__init__(*dependencies)
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

class VectorOperatorDistribution(VectorDistribution):
	def __init__(self, operator, obj, operands):
		super().__init__(obj, *operands)
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

class VectorMethodDistribution(VectorDistribution):
	def __init__(self, method, obj, args):
		super().__init__(*args)
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

class UniformSectorDistribution(VectorDistribution):
	"""Uniform distribution over points in a sector of an annulus"""
	def __init__(self, center, heading, angle, minDist, maxDist):
		super().__init__(center, heading, angle, minDist, maxDist)
		self.center = center
		self.heading = heading
		self.angle = angle
		self.minDist = minDist
		self.maxDist = maxDist

	def sampleGiven(self, value):
		x, y = value[self.center]
		heading, angle, minDist, maxDist = (value[v] for v in
			(self.heading, self.angle, self.minDist, self.maxDist))
		r = random.triangular(minDist, maxDist, maxDist)
		ha = angle / 2.0
		t = random.uniform(-ha, ha) + (heading + (math.pi / 2))
		return Vector(x + (r * cos(t)), y + (r * sin(t)))

	def __str__(self):
		return 'UniformSectorDistribution({}, {}, {}, {}, {})'.format(self.center,
			self.heading, self.angle, self.minDist, self.maxDist)

def needsSampling(thing):
	return isinstance(thing, Distribution) or dependencies(thing)

def scalarOperator(method):
	op = method.__name__
	setattr(VectorDistribution, op, makeOperatorHandler(op))
	def handler2(self, *args):
		if any(needsSampling(arg) for arg in args):
			return MethodDistribution(method, self, args)
		else:
			return method(self, *args)
	return handler2

def makeVectorOperatorHandler(op):
	def handler(self, *args):
		return VectorOperatorDistribution(op, self, args)
	return handler
def vectorOperator(method):
	op = method.__name__
	setattr(VectorDistribution, op, makeVectorOperatorHandler(op))
	def handler2(self, *args):
		if any(needsSampling(arg) for arg in args):
			return VectorMethodDistribution(method, self, args)
		else:
			return method(self, *args)
	return handler2

def vectorDistributionMethod(method):
	def helper(self, *args):
		if any(isinstance(arg, Distribution) for arg in args):
			return VectorMethodDistribution(method, self, args)
		else:
			return method(self, *args)
	return helper

class Vector(Samplable, collections.abc.Sequence):
	def __init__(self, x, y):
		self.coordinates = (x, y)
		super().__init__(self.coordinates)

	@property
	def x(self):
		return self.coordinates[0]

	@property
	def y(self):
		return self.coordinates[1]

	def toVector(self):
		return self

	def sampleGiven(self, value):
		return Vector(*(value[coord] for coord in self.coordinates))

	@vectorOperator
	def rotatedBy(self, angle):
		x, y = self.x, self.y
		c, s = math.cos(angle), math.sin(angle)
		return Vector((c * x) - (s * y), (s * x) + (c * y))

	@vectorOperator
	def offsetRotated(self, heading, offset):
		ro = offset.rotatedBy(heading)
		return self + ro

	@vectorOperator
	def offsetRadially(self, radius, heading):
		return self.offsetRotated(heading, Vector(0, radius))

	@scalarOperator
	def distanceTo(self, other):
		dx, dy = other.toVector() - self
		return math.hypot(dx, dy)

	@scalarOperator
	def angleTo(self, other):
		dx, dy = other.toVector() - self
		return normalizeAngle(math.atan2(dy, dx) - (math.pi / 2))

	@vectorOperator
	def __add__(self, other):
		return Vector(self[0] + other[0], self[1] + other[1])

	@vectorOperator
	def __radd__(self, other):
		return Vector(self[0] + other[0], self[1] + other[1])

	@vectorOperator
	def __sub__(self, other):
		return Vector(self[0] - other[0], self[1] - other[1])

	@vectorOperator
	def __rsub__(self, other):
		return Vector(other[0] - self[0], other[1] - self[1])

	def __len__(self):
		return len(self.coordinates)

	def __getitem__(self, index):
		return self.coordinates[index]

	def __repr__(self):
		return '({} @ {})'.format(self.x, self.y)

VectorDistribution.defaultValueType = Vector

class OrientedVector(Vector):
	def __init__(self, x, y, heading):
		super().__init__(x, y)
		self.heading = heading

# Typing and coercion rules:
#
# coercible to a scalar:
#   float
#   int (by conversion to float)
# coercible to a Vector:
#   anything with a toVector() method
# coercible to an object of type T:
#   instances of T
#
# Finally, Distributions are coercible to T iff their valueType is.

def underlyingType(thing):
	if isinstance(thing, Distribution):
		return thing.valueType
	elif isinstance(thing, TypeChecker) and len(thing.types) == 1:
		return thing.types[0]
	else:
		return type(thing)

def isA(thing, ty):
	return (underlyingType(thing) is ty)

def canCoerceType(typeA, typeB):
	if typeB is float:
		return (typeA is float or typeA is int)
	elif typeB is Vector:
		return hasattr(typeA, 'toVector')
	else:
		return issubclass(typeA, typeB)

def canCoerce(thing, ty):
	tt = underlyingType(thing)
	return canCoerceType(tt, ty)

def coerce(thing, ty):
	assert canCoerce(thing, ty)
	if isinstance(thing, Distribution):
		return thing
	if ty is float:
		return float(thing)
	elif ty is Vector:
		return thing.toVector()
	else:
		return thing

def coerceToAny(thing, types, error):
	for ty in types:
		if canCoerce(thing, ty):
			return coerce(thing, ty)
	raise RuntimeParseError(error)

def toTypes(thing, types, typeError='wrong type'):
	if isinstance(thing, DelayedArgument):
		return TypeChecker(thing, types, typeError)
	else:
		return coerceToAny(thing, types, typeError)

def toType(thing, ty, typeError='wrong type'):
	return toTypes(thing, (ty,), typeError)

def toScalar(thing, typeError='non-scalar in scalar context'):
	return toType(thing, float, typeError)

def toVector(thing, typeError='non-vector in vector context'):
	return toType(thing, Vector, typeError)

def valueRequiringEqualTypes(val, thingA, thingB, typeError='type mismatch'):
	if not isinstance(thingA, DelayedArgument) and not isinstance(thingB, DelayedArgument):
		if underlyingType(thingA) is not underlyingType(thingB):
			raise RuntimeParseError(typeError)
		return val
	else:
		return TypeEqualityChecker(val, thingA, thingB, typeError)

class VectorField(object):
	def __init__(self, name, value):
		super().__init__()
		self.name = name
		self.value = value
		self.valueType = float

	@distributionMethod
	def __getitem__(self, pos):
		return self.value(pos)

	@vectorDistributionMethod
	def followFrom(self, pos, dist, steps=4):
		step = dist / steps
		for i in range(steps):
			pos = pos.offsetRadially(step, self[pos])
		return pos

	def __str__(self):
		return '<VectorField {}>'.format(self.name)

class Region(Samplable):
	def __init__(self, name, *dependencies, orientation=None):
		super().__init__(dependencies)
		self.name = name
		self.orientation = orientation

	def intersect(self, other, triedReversed=False):
		if triedReversed:
			return IntersectionRegion(self, other)
		else:
			return other.intersect(self, triedReversed=True)

	def __str__(self):
		return '<Region {}>'.format(self.name)

class CircularRegion(Region):
	def __init__(self, center, radius):
		super().__init__('Circle', center, radius)
		self.center = center.toVector()
		self.radius = radius
		self.circumcircle = (self.center, self.radius)

	def sampleGiven(self, value):
		return CircularRegion(value[self.center], value[self.radius])

	def __str__(self):
		return 'CircularRegion({},{})'.format(self.center, self.radius)

class SectorRegion(Region):
	def __init__(self, center, radius, heading, angle):
		super().__init__('Sector', center, radius, heading, angle)
		self.center = center.toVector()
		self.radius = radius
		self.heading = heading
		self.angle = angle
		r = (radius / 2) * cos(angle / 2)
		self.circumcircle = (self.center.offsetRadially(r, heading), r)

	def sampleGiven(self, value):
		return SectorRegion(value[self.center], value[self.radius],
			value[self.heading], value[self.angle])

	def containsPoint(self, point):
		point = point.toVector()
		if not pointIsInCone(tuple(point), tuple(self.center), self.heading, self.angle):
			return False
		return point.distanceTo(self.center) <= self.radius

	def uniformPoint(self):
		return UniformSectorDistribution(self.center, self.heading, self.angle, 0, self.radius)

	def __str__(self):
		return 'SectorRegion({},{},{},{})'.format(self.center, self.radius, self.heading, self.angle)

# mixin providing collision detection for rectangular objects and regions
class RotatedRectangle:
	def containsPoint(self, point):
		diff = point - self.position.toVector()
		x, y = diff.rotatedBy(-self.heading)
		return abs(x) <= self.hw and abs(y) <= self.hh

	def intersects(self, rect):
		if not isinstance(rect, RotatedRectangle):
			raise RuntimeError('tried to intersect RotatedRectangle with {}'.format(type(rect)))
		# Quick check by bounding circles
		dx, dy = rect.position.toVector() - self.position.toVector()
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
		base = rectA.position.toVector()
		rot = -rectA.heading
		rc = [(corner - base).rotatedBy(rot) for corner in rectB.corners]
		x, y = zip(*rc)
		minx, maxx = findMinMax(x)
		miny, maxy = findMinMax(y)
		if maxx < -rectA.hw or rectA.hw < minx:
			return True
		if maxy < -rectA.hh or rectA.hh < miny:
			return True
		return False

class RectangularRegion(Region, RotatedRectangle):
	def __init__(self, position, heading, width, height):
		super().__init__('Rectangle', position, heading, width, height)
		self.position = position.toVector()
		self.heading = heading
		self.width = width
		self.height = height
		self.hw = hw = width / 2
		self.hh = hh = height / 2
		self.radius = hypot(hw, hh)		# circumcircle; for collision detection
		self.corners = tuple(position.offsetRotated(heading, offset)
			for offset in ((hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)))
		self.circumcircle = (self.position, self.radius)

	def sampleGiven(self, value):
		return RectangularRegion(value[self.position], value[self.heading],
			value[self.width], value[self.height])

	def uniformPoint(self):
		raise NotImplementedError	# TODO implement

	def __str__(self):
		return 'RectangularRegion({},{},{},{})'.format(self.position, self.heading, self.width, self.height)

class PointSetRegion(Region):
	def __init__(self, name, points, kdTree=None, orientation=None):
		super().__init__(name)
		self.points = points
		self.kdTree = scipy.spatial.cKDTree(points) if kdTree is None else kdTree
		self.orientation = orientation

	def sampleGiven(self, value):
		return self

	def uniformPoint(self):
		return CustomVectorDistribution(lambda values: self.orient(Vector(*random.choice(self.points))),
			name='PointIn({})'.format(self))

	def intersect(self, other, triedReversed=False):
		def sampler(values):
			o = values[other]
			center, radius = o.circumcircle
			possibles = (Vector(*self.kdTree.data[i]) for i in self.kdTree.query_ball_point(center, radius))
			intersection = [p for p in possibles if o.containsPoint(p)]
			if len(intersection) == 0:
				raise RejectionException
			return self.orient(random.choice(intersection))
		return IntersectionRegion(self, other, sampler=sampler, orientation=self.orientation)

	def orient(self, vec):
		if self.orientation is None:
			return vec
		else:
			return OrientedVector(vec.x, vec.y, self.orientation[vec])

class IntersectionRegion(Region):
	def __init__(self, *regions, orientation=None, sampler=None):
		self.regions = list(regions)
		if len(self.regions) < 2:
			raise RuntimeError('tried to take intersection of fewer than 2 regions')
		super().__init__('Intersection', *self.regions, orientation=orientation)
		if sampler is None:
			sampler = self.genericSampler
		self.sampler = sampler

	def containsPoint(self, point):
		return all(region.containsPoint(point) for region in self.regions)

	def uniformPoint(self):
		return CustomVectorDistribution(lambda values: self.sampler(values),
			*self.dependencies, name='PointIn({})'.format(self))

	def genericSampler(self, values):
		regs = [values[reg] for reg in self.regions]
		point = regs[0].uniformPoint().sample()
		for region in regs[1:]:
			if not region.containsPoint(point):
				raise RejectionException
		return point

	def __str__(self):
		return 'IntersectionRegion({})'.format(self.regions)

### Object types

## Support for property defaults

class PropertyDefault:
	def __init__(self, requiredProperties, attributes, value):
		self.requiredProperties = requiredProperties
		self.value = value

		def enabled(thing, default):
			if thing in attributes:
				attributes.remove(thing)
				return True
			else:
				return default
		self.isAdditive = enabled('additive', False)
		for attr in attributes:
			raise RuntimeParseError('unknown property attribute "{}"'.format(attr))

	@staticmethod
	def forValue(value):
		if isinstance(value, PropertyDefault):
			return value
		else:
			return PropertyDefault(set(), set(), lambda self: value)

	def resolveFor(self, prop, overriddenDefs):
		if self.isAdditive:
			allReqs = self.requiredProperties
			for other in overriddenDefs:
				allReqs |= other.requiredProperties
			def concatenator(context):
				allVals = [self.value(context)]
				for other in overriddenDefs:
					allVals.append(other.value(context))
				return tuple(allVals)
			val = DelayedArgument(allReqs, concatenator)
		else:
			val = DelayedArgument(self.requiredProperties, self.value)
		return Specifier(prop, val)

	def makeSpecifier(self):
		pass

## Abstract base class

class Constructable(Samplable):
	"""something created by a constructor"""

	@classmethod
	def defaults(cla):		# TODO improve so this need only be done once?
		# find all defaults provided by the class or its superclasses
		allDefs = collections.defaultdict(list)
		for sc in inspect.getmro(cla):
			if hasattr(sc, '__annotations__'):
				for prop, value in sc.__annotations__.items():
					allDefs[prop].append(PropertyDefault.forValue(value))

		# resolve conflicting defaults
		resolvedDefs = {}
		for prop, defs in allDefs.items():
			primary, rest = defs[0], defs[1:]
			spec = primary.resolveFor(prop, rest)
			resolvedDefs[prop] = spec
		return resolvedDefs

	@classmethod
	def withProperties(cls, props):
		assert all(reqProp in props for reqProp in cls.defaults())
		specs = (Specifier(prop, val) for prop, val in props.items())
		return cls(*specs)

	def __init__(self, *args, **kwargs):
		# Validate specifiers
		specifiers = list(args)
		for prop, val in kwargs.items():
			specifiers.append(Specifier(prop, val))
		properties = dict()
		optionals = collections.defaultdict(list)
		defs = self.defaults()
		for spec in specifiers:
			if not isinstance(spec, Specifier):
				raise RuntimeParseError('argument {} to {} is not a specifier'.format(spec, type(self).__name__))
			prop = spec.property
			if prop in properties:
				raise RuntimeParseError('property "{}" of {} specified twice'.format(prop, type(self).__name__))
			properties[prop] = spec
			for opt in spec.optionals:
				if opt in defs:		# do not apply optionals for properties this object lacks
					optionals[opt].append(spec)

		# Decide which optionals to use
		optionalsForSpec = collections.defaultdict(set)
		for opt, specs in optionals.items():
			if opt in properties:
				continue		# optionals do not override a primary specification
			if len(specs) > 1:
				raise RuntimeParseError('property "{}" of {} specified twice (optionally)'.format(opt, type(self).__name__))
			assert len(specs) == 1
			spec = specs[0]
			properties[opt] = spec
			optionalsForSpec[spec].add(opt)

		# Add any default specifiers needed
		for prop in defs:
			if prop not in properties:
				spec = defs[prop]
				specifiers.append(spec)
				properties[prop] = spec

		# Topologically sort specifiers
		order = []
		seen, done = set(), set()

		def dfs(spec):
			if spec in done:
				return
			elif spec in seen:
				raise RuntimeParseError('specifier for property {} depends on itself'.format(spec.property))
			seen.add(spec)
			for dep in spec.requiredProperties:
				child = properties.get(dep)
				if child is None:
					raise RuntimeParseError('property {} required by specifier {} is not specified'.format(dep, spec))
				else:
					dfs(child)
			order.append(spec)
			done.add(spec)

		for spec in specifiers:
			dfs(spec)
		assert len(order) == len(specifiers)

		# Evaluate and apply specifiers
		for spec in order:
			spec.applyTo(self, optionalsForSpec[spec])

		# Set up dependencies
		deps = []
		for prop in properties:
			assert hasattr(self, prop)
			val = getattr(self, prop)
			if needsSampling(val):
				deps.append(val)
		Samplable.__init__(self, deps)
		self.properties = properties

	def sampleGiven(self, value):
		return self.withProperties({ prop: value[getattr(self, prop)] for prop in self.properties })

	def allProperties(self):
		return { prop: getattr(self, prop) for prop in self.properties }

	def copyWith(self, **overrides):
		props = self.allProperties()
		props.update(overrides)
		return self.withProperties(props)

	def __str__(self):
		if hasattr(self, 'properties'):
			allProps = { prop: getattr(self, prop) for prop in self.properties }
		else:
			allProps = '<under construction>'
		return '{}({})'.format(type(self).__name__, allProps)

