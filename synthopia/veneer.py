
#### VENEER
#### implementations of language constructs

__all__ = (
	# Primitive statements and functions
	'ego', 'require', 'resample', 'param', 'mutate',
	'sin', 'cos', 'hypot', 'max', 'min',
	# Prefix operators
	'Visible',
	'Front', 'Back', 'Left', 'Right',
	'FrontLeft', 'FrontRight', 'BackLeft', 'BackRight',
	# Infix operators
	'FieldAt', 'RelativeTo', 'OffsetAlong', 'RelativePosition',
	'RelativeHeading', 'ApparentHeading',
	'DistanceFrom', 'Follow', 'CanSee',
	# Primitive types
	'Vector', 'Region', 'PointSetRegion', 'VectorField', 'Mutator',
	'Range', 'Options',
	# Constructible types and specifiers
	'Point', 'With', 'At', 'In', 'Beyond', 'VisibleFrom', 'VisibleSpec', 'OffsetBy',
	'OrientedPoint', 'Facing', 'FacingToward', 'ApparentlyFacing',
	'Object', 'LeftSpec', 'RightSpec', 'Ahead',
	# Temporary stuff... # TODO remove
	'PropertyDefault'
)

# everything that should not be directly accessible from the language is imported here:
from midlayer import *
from scenarios import CarModel, CarColor, Options, Map		# TODO rejigger?

### Primitive statements and functions

egoObject = None

def ego(obj=None):
	global egoObject
	if obj is None:
		if egoObject is None:
			raise RuntimeParseError('referred to ego object not yet assigned')
	elif not isinstance(obj, Object):
		raise RuntimeParseError('tried to make non-object the ego object')
	else:
		egoObject = obj
	return egoObject

allRequirements = {}

def require(req, line, prob=1):
	# the translator wrapped the requirement in a lambda to prevent evaluation,
	# so we need to save the current values of all referenced names; throw in
	# the ego object too since it can be referred to implicitly
	allRequirements[req] = (getAllGlobals(req), egoObject, line, prob)

def getAllGlobals(req, restrictTo=None):
	namespace = req.__globals__
	if restrictTo is not None and restrictTo is not namespace:
		return {}
	externals = inspect.getclosurevars(req)
	assert not externals.nonlocals		# TODO handle these
	globs = dict(externals.builtins)
	for name, value in externals.globals.items():
		globs[name] = value
		if inspect.isfunction(value):
			subglobs = getAllGlobals(value, restrictTo=namespace)
			for name, value in subglobs.items():
				if name in globs:
					assert value is globs[name]
				else:
					globs[name] = value
	return globs

def resample(dist):
	return dist.clone() if isinstance(dist, Distribution) else dist

globalParameters = {}

def param(**params):
	globalParameters.update(params)

def mutate(*objects):
	if len(objects) == 0:
		objects = allObjects
	for obj in objects:
		if not isinstance(obj, Object):
			raise RuntimeParseError('"mutate X" with X not an object')
		obj.mutationEnabled = True

### Prefix operators

# visible <region>
def Visible(region):
	if not isinstance(region, Region):
		raise RuntimeParseError('"visible X" with X not a Region')
	return region.intersect(ego().visibleRegion)

# front of <object>, etc.
ops = (
	'front', 'back', 'left', 'right',
	'front left', 'front right',
	'back left', 'back right'
)
template = """\
def {function}(X):
	if not isinstance(X, Object):
		raise RuntimeParseError('"{syntax} of X" with X not an Object')
	return X.{property}
"""
for op in ops:
	func = ''.join(word.capitalize() for word in op.split(' '))
	prop = func[0].lower() + func[1:]
	definition = template.format(function=func, syntax=op, property=prop)
	exec(definition)

### Infix operators

# <field> at <vector>
def FieldAt(X, Y):
	if not isinstance(X, VectorField):
		raise RuntimeParseError('"X at Y" with X not a vector field')
	Y = toVector(Y, '"X at Y" with Y not a vector')
	return X[Y]

# F relative to G (with at least one of F, G a field, the other a field or heading)
# <vector> relative to <oriented point> (and vice versa)
# <vector> relative to <vector>
# <heading> relative to <heading>
def RelativeTo(X, Y):
	xf, yf = isA(X, VectorField), isA(Y, VectorField)
	if xf or yf:
		if xf and yf and X.valueType != Y.valueType:
			raise RuntimeParseError('"X relative to Y" with X, Y fields of different types')
		fieldType = X.valueType if xf else Y.valueType
		error = '"X relative to Y" with field and value of different types'
		def helper(context):
			pos = context.position.toVector()
			xp = X[pos] if xf else toType(X, fieldType, error)
			yp = Y[pos] if yf else toType(Y, fieldType, error)
			return xp + yp
		return DelayedArgument({'position'}, helper)
	else:
		if isinstance(X, OrientedPoint):	# TODO too strict?
			if isinstance(Y, OrientedPoint):
				raise RuntimeParseError('"X relative to Y" with X, Y both oriented points')
			Y = toVector(Y, '"X relative to Y" with X an oriented point but Y not a vector')
			return X.relativize(Y)
		elif isinstance(Y, OrientedPoint):
			X = toVector(X, '"X relative to Y" with Y an oriented point but X not a vector')
			return Y.relativize(X)
		else:
			X = toTypes(X, (Vector, float), '"X relative to Y" with X neither a vector nor scalar')
			Y = toTypes(Y, (Vector, float), '"X relative to Y" with Y neither a vector nor scalar')
			return valueRequiringEqualTypes(X + Y, X, Y, '"X relative to Y" with vector and scalar')

# <vector> offset along <heading> by <vector>
# <vector> offset along <field> by <vector>
def OffsetAlong(X, H, Y):
	X = toVector(X, '"X offset along H by Y" with X not a vector')
	Y = toVector(Y, '"X offset along H by Y" with Y not a vector')
	if isinstance(H, VectorField):
		H = H[X]
	H = toScalar(H, '"X offset along H by Y" with H not a heading or vector field')
	return X.offsetRotated(H, Y)

# relative position of <vector> from <vector>
def RelativePosition(X, Y=None):
	X = toVector(X, '"relative position of X from Y" with X not a vector')
	if Y is None:
		Y = ego()
	Y = toVector(Y, '"relative position of X from Y" with Y not a vector')
	return X - Y

# relative heading of <oriented point> from <oriented point>
def RelativeHeading(X, Y=None):
	if not isinstance(X, OrientedPoint):
		raise RuntimeParseError('"relative heading of X from Y" with X not an OrientedPoint')
	if Y is None:
		Y = ego()
	elif not isinstance(Y, OrientedPoint):
		raise RuntimeParseError('"relative heading of X from Y" with Y not an OrientedPoint')
	return normalizeAngle(X.heading - Y.heading)

# apparent heading of <oriented point> from <vector>
def ApparentHeading(X, Y=None):
	if not isinstance(X, OrientedPoint):
		raise RuntimeParseError('"apparent heading of X from Y" with X not an OrientedPoint')
	if Y is None:
		Y = ego()
	Y = toVector(Y, '"relative heading of X from Y" with Y not a vector')
	return apparentHeadingAtPoint(X.position, X.heading, Y)

# distance from <vector> to <vector>
def DistanceFrom(X, Y=None):
	X = toVector(X, '"distance from X to Y" with X not a vector')
	if Y is None:
		Y = ego()
	Y = toVector(Y, '"distance from X to Y" with Y not a vector')
	return X.distanceTo(Y)

# follow <field> from <vector> for <number>
def Follow(F, X, D):
	if not isinstance(F, VectorField):
		raise RuntimeParseError('"follow F from X for D" with F not a vector field')
	X = toVector(X, '"follow F from X for D" with X not a vector')
	D = toScalar(D, '"follow F from X for D" with D not a number')
	pos = F.followFrom(X, D)
	heading = F[pos]
	return OrientedPoint(position=pos, heading=heading)

# <object> can see <vector>		# TODO on object, check all corners!
def CanSee(X, Y):
	if not isinstance(X, Object):
		raise RuntimeParseError('"X can see Y" with X not an Object')
	Y = toVector(Y, '"X can see Y" with Y not a vector')
	return X.visibleRegion.containsPoint(Y)

### Mutators

class Mutator:
	pass

class PositionMutator(Mutator):
	def __init__(self, stddev):
		self.stddev = stddev

	def appliedTo(self, obj):
		noise = Vector(random.gauss(0, self.stddev), random.gauss(0, self.stddev))
		pos = toVector(obj.position, '"position" not a vector')
		pos = pos + noise
		return (obj.copyWith(position=pos), True)		# allow further mutation

class HeadingMutator(Mutator):
	def __init__(self, stddev):
		self.stddev = stddev

	def appliedTo(self, obj):
		noise = random.gauss(0, self.stddev)
		h = obj.heading + noise
		return (obj.copyWith(heading=h), True)		# allow further mutation

### Object types and specifiers

## Point

class Point(Constructable):
	position: Vector(0, 0)
	visibleDistance: 50

	mutationEnabled: False
	mutator: PropertyDefault({'positionStdDev'}, {'additive'}, lambda self: PositionMutator(self.positionStdDev))
	positionStdDev: 1

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.corners = (self.position,)
		self.visibleRegion = CircularRegion(self.position, self.visibleDistance)

	def toVector(self):
		return self.position.toVector()

	def canSee(self, other):
		for corner in other.corners:
			if self.distanceTo(corner) <= self.visibleDistance:
				return True
		return False

	def sampleGiven(self, value):
		sample = super().sampleGiven(value)
		if self.mutationEnabled:
			for mutator in self.mutator:
				if mutator is None:
					continue
				sample, proceed = mutator.appliedTo(sample)
				if not proceed:
					break
		return sample

	# Points automatically convert to Vectors when needed
	def __getattr__(self, attr):
		if hasattr(Vector, attr):
			return getattr(self.toVector(), attr)
		else:
			raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

# with <property> <value>
def With(prop, val):
	return Specifier(prop, val)

# at <vector>
def At(pos):
	pos = toVector(pos, 'specifier "at X" with X not a vector')
	return Specifier('position', pos)

# in/on <region>
def In(region):
	if not isinstance(region, Region):
		raise RuntimeParseError('specifier "in/on R" with R not a Region')
	extras = {} if region.orientation is None else {'heading'}
	return Specifier('position', region.uniformPoint(), optionals=extras)

# beyond <vector> by <vector> from <vector>
def Beyond(pos, offset, fromPt=None):
	pos = toVector(pos, 'specifier "beyond X by Y" with X not a vector')
	offset = toVector(offset, 'specifier "beyond X by Y" with Y not a vector')
	if fromPt is None:
		fromPt = ego()
	fromPt = toVector(fromPt, 'specifier "beyond X by Y from Z" with Z not a vector')
	lineOfSight = fromPt.angleTo(pos)
	return Specifier('position', pos.offsetRotated(lineOfSight, offset))

# visible from <Object>
def VisibleFrom(obj):
	if not isinstance(obj, Object):
		raise RuntimeParseError('specifier "visible from O" with O not an Object')
	return Specifier('position', obj.visibleRegion.uniformPoint())

# visible
def VisibleSpec():
	return VisibleFrom(ego())

# offset by <vector>
def OffsetBy(offset):
	offset = toVector(offset, 'specifier "offset by X" with X not a vector')
	pos = RelativeTo(offset, ego()).toVector()
	return Specifier('position', pos)

## OrientedPoint

class OrientedPoint(Point):
	heading: 0
	viewAngle: math.tau

	mutator: PropertyDefault({'headingStdDev'}, {'additive'},
		lambda self: HeadingMutator(self.headingStdDev))
	headingStdDev: math.radians(5)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.visibleRegion = SectorRegion(self.position, self.visibleDistance, self.heading, self.viewAngle)

	def relativize(self, vec):
		pos = self.relativePosition(vec)
		return OrientedPoint(position=pos, heading=self.heading)

	def relativePosition(self, x, y=None):
		vec = x if y is None else Vector(x, y)
		pos = self.position.offsetRotated(self.heading, vec)
		return OrientedPoint(position=pos, heading=self.heading)

	def canSee(self, other):	# TODO fix this approximation!
		pos = self.position.toVector()
		for corner in other.corners:
			if (pointIsInCone(corner, pos, self.heading, self.viewAngle)
				and pos.distanceTo(corner) <= self.visibleDistance):
				return True
		return False

# facing <field>
# facing <number>
def Facing(heading):
	if isinstance(heading, VectorField):
		return Specifier('heading', DelayedArgument({'position'}, lambda self: heading[self.position]))
	else:
		heading = toScalar(heading, 'specifier "facing X" with X not a number or vector field')
		return Specifier('heading', heading)

# facing toward <vector>
def FacingToward(pos):
	pos = toVector(pos, 'specifier "facing toward X" with X not a vector')
	return Specifier('heading', DelayedArgument({'position'}, lambda self: self.position.angleTo(pos)))

# apparently facing <number> from <vector>
def ApparentlyFacing(heading, fromPt=None):
	heading = toScalar(heading, 'specifier "apparently facing X" with X not a number')
	if fromPt is None:
		fromPt = ego()
	fromPt = toVector(fromPt, 'specifier "apparently facing X from Y" with Y not a vector')
	return Specifier('heading', DelayedArgument({'position'}, lambda self: fromPt.angleTo(self.position) + heading))

## Object

allObjects = set()

class Object(OrientedPoint, RotatedRectangle):
	width: 1
	height: 1
	allowCollisions: False
	requireVisible: True

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		allObjects.add(self)
		self.hw = hw = self.width / 2
		self.hh = hh = self.height / 2
		self.radius = hypot(hw, hh)	# circumcircle; for collision detection
		self.left = self.relativePosition(-hw, 0)
		self.right = self.relativePosition(hw, 0)
		self.front = self.relativePosition(0, hh)
		self.back = self.relativePosition(0, -hh)
		self.frontLeft = self.relativePosition(-hw, hh)
		self.frontRight = self.relativePosition(hw, hh)
		self.backLeft = self.relativePosition(-hw, -hh)
		self.backRight = self.relativePosition(hw, -hh)
		self.corners = (self.frontRight.toVector(), self.frontLeft.toVector(),
			self.backLeft.toVector(), self.backRight.toVector())

	def show(self, map, plt, highlight=False):
		if dependencies(self):
			raise RuntimeError('tried to show() symbolic Object')
		pos = self.position.toVector()
		mpos = map.langToMapCoords(pos)

		if highlight:
			# Circle around car
			rad = 1.5 * max(self.width, self.height)
			c = plt.Circle(mpos, rad, color='g', fill=False)
			plt.gca().add_artist(c)
			# View cone
			ha = self.viewAngle / 2.0
			for angle in (-ha, ha):
				p = pos.offsetRadially(20, self.heading + angle)
				edge = [mpos, map.langToMapCoords(p)]
				x, y = zip(*edge)
				plt.plot(x, y, 'b:')

		corners = [map.langToMapCoords(corner) for corner in self.corners]
		x, y = zip(*corners)
		plt.fill(x, y, color=self.color)
		plt.plot(x + (x[0],), y + (y[0],), color="w", linewidth=1)

		frontMid = averageVectors(corners[0], corners[1])
		baseTriangle = [frontMid, corners[2], corners[3]]
		triangle = [averageVectors(p, mpos, weight=0.5) for p in baseTriangle]
		x, y = zip(*triangle)
		plt.fill(x, y, "w")
		plt.plot(x + (x[0],), y + (y[0],), color="k", linewidth=1)

# left of <oriented point>
# left of <vector>
def LeftSpec(pos):
	extras = set()
	if isinstance(pos, OrientedPoint):		# TODO too strict?
		new = DelayedArgument({'width'}, lambda self: pos.relativePosition(-self.width / 2, 0))
		extras.add('heading')
	else:
		pos = toVector(pos, 'specifier "left of X" with X not a vector')
		new = DelayedArgument({'width', 'heading'},
					lambda self: pos.offsetRotated(self.heading, Vector(-self.width / 2, 0)))
	return Specifier('position', new, optionals=extras)

# right of <oriented point>
# right of <vector>
def RightSpec(pos):
	extras = set()
	if isinstance(pos, OrientedPoint):		# TODO too strict?
		new = DelayedArgument({'width'}, lambda self: pos.relativePosition(self.width / 2, 0))
		extras.add('heading')
	else:
		pos = toVector(pos, 'specifier "right of X" with X not a vector')
		new = DelayedArgument({'width', 'heading'},
					lambda self: pos.offsetRotated(self.heading, Vector(self.width / 2, 0)))
	return Specifier('position', new, optionals=extras)

# ahead of <oriented point>
# ahead of <vector>
def Ahead(pos):
	extras = set()
	if isinstance(pos, OrientedPoint):		# TODO too strict?
		new = DelayedArgument({'height'}, lambda self: pos.relativePosition(0, self.height / 2))
		extras.add('heading')
	else:
		pos = toVector(pos, 'specifier "ahead of X" with X not a vector')
		new = DelayedArgument({'height', 'heading'},
					lambda self: pos.offsetRotated(self.heading, Vector(0, self.height / 2)))
	return Specifier('position', new, optionals=extras)
