
#### TRANSLATOR
#### turns a scene description into a Scenario object

import sys
import os
import traceback
import time
import inspect
import types
import importlib
import importlib.abc
import importlib.util
from collections import namedtuple

import tokenize
from tokenize import NAME, NL, NEWLINE, ENDMARKER, OP, NUMBER, COLON, COMMENT, ENCODING
from tokenize import LPAR, RPAR, LSQB, RSQB, COMMA, DOUBLESLASH, DOUBLESLASHEQUAL
from tokenize import AT, LEFTSHIFT, RIGHTSHIFT, VBAR, AMPER, TILDE, CIRCUMFLEX, STAR
from tokenize import LEFTSHIFTEQUAL, RIGHTSHIFTEQUAL, VBAREQUAL, AMPEREQUAL, CIRCUMFLEXEQUAL
from tokenize import INDENT, DEDENT

import ast
from ast import parse, dump, NodeVisitor, NodeTransformer, copy_location, fix_missing_locations
from ast import Load, Store, Name, Call, Tuple, BinOp, MatMult, BitAnd, BitOr, BitXor, LShift
from ast import RShift, Starred, Lambda, AnnAssign, Set, Str, Num, Subscript, Index

from scenarios import Scenario, Samplable
from veneer import Constructable

### THE TOP LEVEL: loading scene DSL modules

dslExtensions = ('sc', 'scene')

class DSLMetaFinder(importlib.abc.MetaPathFinder):
	def find_spec(self, name, path, target):
		if path is not None:
			return None			# we don't handle submodules
		path = os.getcwd()
		for extension in dslExtensions:
			filename = name + '.' + extension
			filepath = os.path.join(path, filename)
			if os.path.exists(filepath):
				spec = importlib.util.spec_from_file_location(name, filepath,
					loader=DSLLoader(filepath, filename))
				return spec
		return None

class DSLLoader(importlib.abc.Loader):
	def __init__(self, filepath, filename):
		self.filepath = filepath
		self.filename = filename

	def create_module(self, spec):
		return None

	def exec_module(self, module):
		filename = self.filename
		namespace = module.__dict__
		# Read source file
		with open(self.filepath, 'rb') as f:
			encoding, sourceLines = tokenize.detect_encoding(f.readline)
			sourceLines += f.readlines()
		# We first execute all imports in case they import constructors
		afterImportBlock = executeImportsIn(sourceLines, namespace, encoding, filename)
		importedConstructors = findConstructorsIn(namespace)
		# Translate token stream to valid Python syntax
		tokens = tokenize.tokenize(iter(sourceLines[afterImportBlock:]).__next__)
		translator = TokenTranslator(importedConstructors)
		newSource, module._lineMap, allConstructors = translator.translate(tokens)
		# Parse the translated source
		tree = parseTranslatedSource(newSource, module, filename=filename)
		# Modify the parse tree to produce the correct semantics
		newTree = translateParseTree(tree, module, allConstructors)
		# Compile the final code and execute it
		executeTranslatedTreeIn(newTree, namespace, module, filename=filename)
		# Extract scenario state
		extractScenarioStateFrom(module, namespace, filename)

# register finder for scenario DSL files
sys.meta_path.insert(0, DSLMetaFinder())

### TRANSLATION PHASE ZERO: definitions of language elements not already in Python

## Options

showInternalBacktrace = True	# put False to suppress potentially-misleading info

## Get Python names of various elements
## (for checking consistency between the translator and the veneer)

import veneer		# TODO better way to get this?
api = set(veneer.__all__)
del veneer

## Preamble
# (included at the beginning of every module to be translated;
# the first line imports a secret reference to the veneer so we can extract
# internal state after running the translated code; the second line imports
# the implementations of the public language features)
veneerReference = '_veneer'
preamble = \
"""\
import veneer as {veneerRef}
from veneer import *
""".format(veneerRef=veneerReference)
preambleLines = preamble.count('\n')	# for correcting line numbers in error messages

## Functions used internally

rangeConstructor = Name('Range', Load())
createDefault = Name('PropertyDefault', Load())
internalFunctions = { rangeConstructor.id, createDefault.id }

# sanity check: these functions actually exist
for imp in internalFunctions:
	assert imp in api, imp

## Statements implemented by functions

requireStatement = 'require'
functionStatements = { requireStatement, 'param', 'mutate' }

# sanity check: implementations of statements actually exist
for imp in functionStatements:
	assert imp in api, imp

## Built-in functions

builtinFunctions = { 'resample' }

# sanity check: implementations of built-in functions actually exist
for imp in builtinFunctions:
	assert imp in api, imp

## Constructors and specifiers

constructorStatement = 'constructor'	# statement defining a new constructor
Constructor = namedtuple('Constructor', ('name', 'parent', 'specifiers'))

pointSpecifiers = {
	('visible', 'from'): 'VisibleFrom',
	('offset', 'by'): 'OffsetBy',
	('at',): 'At',
	('in',): 'In',
	('on',): 'In',
	('beyond',): 'Beyond',
	('visible',): 'VisibleSpec'
}
orientedPointSpecifiers = {
	('apparently', 'facing'): 'ApparentlyFacing',
	('facing', 'toward'): 'FacingToward',
	('facing',): 'Facing'
}
objectSpecifiers = {
	('left', 'of'): 'LeftSpec',
	('right', 'of'): 'RightSpec',		# TODO implement
	('ahead', 'of'): 'Ahead'
#	('behind',): 'Behind'
}

# sanity check: implementations of specifiers actually exist
for imp in pointSpecifiers.values():
	assert imp in api, imp
for imp in objectSpecifiers.values():
	assert imp in api, imp

builtinConstructors = {
	'Point': Constructor('Point', None, pointSpecifiers),
	'OrientedPoint': Constructor('OrientedPoint', 'Point', orientedPointSpecifiers),
	'Object': Constructor('Object', 'OrientedPoint', objectSpecifiers)
}
functionStatements.update(builtinConstructors)

# sanity check: built-in constructors actually exist
for const in builtinConstructors:
	assert const in api, const

## Prefix operators

prefixOperators = {
	('relative', 'position'): 'RelativePosition',
	('relative', 'heading'): 'RelativeHeading',
	('apparent', 'heading'): 'ApparentHeading',
	('distance', 'from'): 'DistanceFrom',
	('distance', 'to'): 'DistanceFrom',
	('ego', '='): 'ego',
	('front', 'left'): 'FrontLeft',
	('front', 'right'): 'FrontRight',
	('back', 'left'): 'BackLeft',
	('back', 'right'): 'BackRight',
	('front',): 'Front',
	('back',): 'Back',
	('left',): 'Left',
	('right',): 'Right',
	('follow',): 'Follow',
	('visible',): 'Visible'
}
assert all(1 <= len(op) <= 2 for op in prefixOperators)
prefixIncipits = { op[0] for op in prefixOperators }
assert not any(op in functionStatements for op in prefixIncipits)

# sanity check: implementations of prefix operators actually exist
for imp in prefixOperators.values():
	assert imp in api, imp

## Infix operators

# pseudo-operator for encoding argument packages for (3+)-ary ops
packageToken = (RIGHTSHIFT, '>>')
packageNode = RShift

InfixOp = namedtuple('InfixOp', ('syntax', 'implementation', 'arity', 'token', 'node'))
infixOperators = (
	# existing Python operators with new semantics
	InfixOp('@', 'Vector', 2, None, MatMult),

	# operators not in Python (in decreasing precedence order)
	InfixOp('at', 'FieldAt', 2, (LEFTSHIFT, '<<'), LShift),
	InfixOp('relative to', 'RelativeTo', 2, (AMPER, '&'), BitAnd),
	InfixOp('offset by', 'RelativeTo', 2, (AMPER, '&'), BitAnd),
	InfixOp('offset along', 'OffsetAlong', 3, (CIRCUMFLEX, '^'), BitXor),
	InfixOp('can see', 'CanSee', 2, (VBAR, '|'), BitOr),

	# just syntactic conveniences, not really operators
	InfixOp('from', None, 2, (COMMA, ','), None),
	InfixOp('for', None, 2, (COMMA, ','), None),
	InfixOp('to', None, 2, (COMMA, ','), None),
	InfixOp('by', None, 2, packageToken, None)
)

infixTokens = {}
infixImplementations = {}
infixIncipits = set()
for op in infixOperators:
	# if necessary, set up map from language to Python syntax
	if op.token is not None:
		tokens = tuple(op.syntax.split(' '))
		assert 1 <= len(tokens) <= 2, op
		assert tokens not in infixTokens, op
		infixTokens[tokens] = op.token
		incipit = tokens[0]
		assert incipit not in functionStatements, op
		infixIncipits.add(incipit)
	# if necessary, set up map from Python to language semantics
	imp = op.implementation
	if imp is not None:
		assert imp in api, op
		node = op.node
		if node in infixImplementations:	# two operators may have the same implementation
			oldArity, oldName = infixImplementations[node]
			assert op.arity == oldArity, (op, oldName.id)
			assert imp == oldName.id, (op, oldName.id)
		else:
			infixImplementations[node] = (op.arity, Name(imp, Load()))

allIncipits = prefixIncipits | infixIncipits

## Direct syntax replacements

replacements = {	# TODO police the usage of these? could yield bizarre error messages
	'of': tuple(),
	'deg': ((STAR, '*'), (NUMBER, '0.01745329252')),
	'ego': ((NAME, 'ego'), (LPAR, '('), (RPAR, ')'))
}

## Illegal and reserved syntax

illegalTokens = {
	LEFTSHIFT, RIGHTSHIFT, VBAR, AMPER, TILDE, CIRCUMFLEX,
	LEFTSHIFTEQUAL, RIGHTSHIFTEQUAL, VBAREQUAL, AMPEREQUAL, CIRCUMFLEXEQUAL,
	DOUBLESLASH, DOUBLESLASHEQUAL
}

# sanity check: stand-in tokens for infix operators must be illegal
for token in infixTokens.values():
	ttype = token[0]
	assert (ttype is COMMA or ttype in illegalTokens), token

keywords = {constructorStatement, 'import'} \
	| internalFunctions | functionStatements \
	| prefixIncipits | infixIncipits \
	| replacements.keys()

### TRANSLATION PHASE ONE: handling imports

class ParseError(Exception):
	def __init__(self, tokenOrLine, message):
		line = tokenOrLine.start[0] if hasattr(tokenOrLine, 'start') else tokenOrLine
		super().__init__('Parse error in line ' + str(line) + ': ' + message)

def executeImportsIn(sourceLines, namespace, encoding, filename):
	"""Because constructor definitions change the way subsequent tokens are
	 transformed, and such definitions are importable, we must process all
	 imports early (rather than during Python execution, which would be more
	 natural). For simplicity we require all imports to be at the beginning of
	 the source file.
	 """
	# Find block of imports at beginning of file
	importBlock = []
	afterBlock = None
	for number, line in enumerate(sourceLines):
		line = str(line, encoding=encoding)
		commentStart = line.find('#')
		if commentStart >= 0:
			line = line[:commentStart]
		if afterBlock is None:
			if 'import' in line or line.strip() == '':
				importBlock.append(line)
			else:
				afterBlock = number
		elif 'import' in line:
			raise ParseError(number + 1, 'import not at beginning of file')
	if afterBlock is None:
		afterBlock = len(sourceLines)

	# Execute imports
	source = '\n'.join(importBlock)
	try:
		exec(compile(source, filename, 'exec'), namespace)
	except Exception as e:
		cause = e if showInternalBacktrace else None
		raise InterpreterParseError(e, None, filename) from cause

	return afterBlock

def findConstructorsIn(namespace):
	constructors = []
	for name, value in namespace.items():
		if inspect.isclass(value) and issubclass(value, Constructable):
			if name in builtinConstructors:
				continue
			parent = None
			for base in value.__bases__:
				if issubclass(base, Constructable):
					assert parent is None
					parent = base
			constructors.append(Constructor(name, parent.__name__, {}))
	return constructors

### TRANSLATION PHASE TWO: translation at the level of tokens

# utility class to allow iterator lookahead
class Peekable:
	def __init__(self, gen):
		self.gen = iter(gen)
		self.current = next(self.gen, None)

	def __iter__(self):
		return self

	def __next__(self):
		cur = self.current
		if cur is None:
			raise StopIteration
		self.current = next(self.gen, None)
		return cur

	def peek(self):
		return self.current

def peek(thing):
	return thing.peek()

class TokenTranslator:
	def __init__(self, constructors=()):
		self.functions = set(functionStatements)
		self.constructors = dict(builtinConstructors)
		for constructor in constructors:
			name = constructor.name
			assert name not in self.constructors
			self.constructors[name] = constructor
			self.functions.add(name)

	def createConstructor(self, name, parent, specs={}):
		if parent is None:
			parent = 'Object'		# default superclass
		self.constructors[name] = Constructor(name, parent, specs)
		self.functions.add(name)
		return parent

	def specifiersForConstructor(self, const):
		name, parent, specs = self.constructors[const]
		if parent is None:
			return specs
		else:
			ps = dict(self.specifiersForConstructor(parent))
			ps.update(specs)
			return ps

	def translate(self, tokens):
		"""Process the token stream, adding or modifying tokens as necessary to
		 produce valid Python syntax."""
		tokens = Peekable(tokens)
		newTokens = []
		functionStack = []
		inConstructor = False	# inside a constructor or one of its specifiers
		parenLevel = 0
		lineCount = 0
		lastLine = 0
		lineMap = { 0: 0 }
		startOfLine = True		# TODO improve hack?
		functions = self.functions
		constructors = self.constructors
		for token in tokens:
			ttype = token.exact_type
			tstring = token.string
			skip = False

			# Catch Python operators that can't be used in the language
			if ttype in illegalTokens:
				raise ParseError(token, 'illegal operator "{}"'.format(tstring))

			# Determine which operators are allowed in current context
			context, startLevel = functionStack[-1] if functionStack else (None, None)
			inConstructorContext = (context in constructors and parenLevel == startLevel)
			if inConstructorContext:
				inConstructor = True
				allowedPrefixOps = self.specifiersForConstructor(context)
				allowedInfixOps = dict()
			else:
				allowedPrefixOps = prefixOperators
				allowedInfixOps = infixTokens

			# Parse next token
			if ttype == LPAR or ttype == LSQB:		# keep track of nesting level
				parenLevel += 1
			elif ttype == RPAR or ttype == RSQB:	# ditto
				parenLevel -= 1
			elif ttype in (NEWLINE, NL, ENDMARKER):	# track non-logical lines for error reporting
				lineCount += 1
				lineMap[lineCount] = lastLine + 1
				lastLine = token.start[0]
			elif ttype == NAME:		# the interesting case: all new syntax falls in here
				function = None
				argument = None

				# try to match 2-word language constructs
				matched = False
				nextToken = peek(tokens)		# lookahead so we can give 2-word ops precedence
				if nextToken is not None:
					nextString = nextToken.string
					twoWords = (tstring, nextString)
					if startOfLine and tstring == 'for':	# TODO improve hack?
						matched = True
					elif startOfLine and tstring == constructorStatement:	# constructor definition
						if nextToken.type != NAME or nextString in keywords:
							raise ParseError(nextToken, 'invalid constructor name "{}"'.format(nextString))
						next(tokens)	# consume name
						parent = None
						if peek(tokens).exact_type == LPAR:		# superclass specification
							next(tokens)
							parentToken = next(tokens)
							parent = parentToken.string
							if parentToken.exact_type != NAME or parent in keywords:
								raise ParseError(parentToken, 'invalid constructor superclass "{}"'.format(parent))
							if parent not in self.constructors:
								raise ParseError(parentToken, 'constructor cannot subclass non-object "{}"'.format(parent))
							if next(tokens).exact_type != RPAR:
								raise ParseError(parentToken, 'malformed constructor definition')
						if peek(tokens).exact_type != COLON:
							raise ParseError(nextToken, 'malformed constructor definition')
						parent = self.createConstructor(nextString, parent)
						newTokens.append((NAME, 'class'))
						newTokens.append((NAME, nextString))
						newTokens.append((LPAR, '('))
						newTokens.append((NAME, parent))
						newTokens.append((RPAR, ')'))
						skip = True
						matched = True
					elif twoWords in allowedPrefixOps:	# 2-word prefix operator
						function = allowedPrefixOps[twoWords]
						next(tokens)	# consume second word
						matched = True
					elif not startOfLine and twoWords in allowedInfixOps:	# 2-word infix operator
						newTokens.append(allowedInfixOps[twoWords])
						next(tokens)
						skip = True
						matched = True
					elif inConstructorContext and tstring == 'with':	# special case for 'with' specifier
						function = 'With'
						argument = '"' + nextString + '"'
						next(tokens)
						matched = True
					elif tstring == requireStatement and nextString == '[':		# special case for require[p]
						next(tokens)	# consume '['
						prob = next(tokens)
						if prob.exact_type != NUMBER:
							raise ParseError(prob, 'soft requirement must have constant probability')
						if next(tokens).exact_type != RSQB:
							raise ParseError(prob, 'malformed soft requirement')
						function = requireStatement
						argument = prob.string
						matched = True
				if not matched:
					# 2-word constructs don't match; try 1-word
					oneWord = (tstring,)
					if oneWord in allowedPrefixOps:		# 1-word prefix operator
						function = allowedPrefixOps[oneWord]
					elif not startOfLine and oneWord in allowedInfixOps:	# 1-word infix operator
						newTokens.append(allowedInfixOps[oneWord])
						skip = True
					elif inConstructorContext:		# couldn't match any 1- or 2-word specifier
						raise ParseError(token, 'unknown constructor specifier "{}"'.format(tstring))
					elif tstring in functions:		# built-in function
						function = tstring
					elif tstring in replacements:	# direct replacement
						newTokens.extend(replacements[tstring])
						skip = True
					elif startOfLine and tstring == 'from':		# special case to allow 'from X import Y'
						pass
					elif tstring in keywords:		# some malformed usage
						raise ParseError(token, 'unexpected keyword "{}"'.format(tstring))
					else:
						pass	# nothing matched; pass through unchanged to Python

				# generate new tokens for function calls
				if function is not None:
					functionStack.append((function, parenLevel))
					newTokens.append((NAME, function))
					newTokens.append((LPAR, '('))
					if argument is not None:
						newTokens.append((NAME, argument))
						newTokens.append((COMMA, ','))
					skip = True

			# Detect the end of function argument lists
			if len(functionStack) > 0:
				context, startLevel = functionStack[-1]
				while parenLevel < startLevel:		# we've closed all parens for the current function
					functionStack.pop()
					newTokens.append((RPAR, ')'))
					context, startLevel = (None, 0) if len(functionStack) == 0 else functionStack[-1]
				if inConstructor and parenLevel == startLevel and ttype == COMMA:		# starting a new specifier
					while functionStack and context not in constructors:
						functionStack.pop()
						newTokens.append((RPAR, ')'))
						context, startLevel = (None, 0) if len(functionStack) == 0 else functionStack[-1]
				elif ttype == NEWLINE or ttype == ENDMARKER or ttype == COMMENT:	# end of line
					inConstructor = False
					if parenLevel != 0:
						raise ParseError(token, 'unmatched parens/brackets')
					while len(functionStack) > 0:
						functionStack.pop()
						newTokens.append((RPAR, ')'))

			# Output token unchanged, unless handled above
			if not skip:
				token = token[:2]	# hack to get around bug in untokenize
				newTokens.append(token)
			startOfLine = (ttype in (ENCODING, NEWLINE, NL, INDENT, DEDENT))

		rewrittenSource = preamble + str(tokenize.untokenize(newTokens), encoding='utf-8')
		return (rewrittenSource, lineMap, self.constructors)

def originalSourceLine(line, module):
	return module._lineMap[line - preambleLines]

### TRANSLATION PHASE THREE: parsing of Python resulting from token translation

class PythonParseError(Exception):
	def __init__(self, syntaxError, line):
		super().__init__('Parse error in line ' + str(line) + ': ' + syntaxError.msg)

def parseTranslatedSource(source, module, filename='<autogenerated Python>'):
	try:
		tree = parse(source, filename=filename)
		#print(dump(tree))
		return tree
	except SyntaxError as e:
		raise PythonParseError(e, originalSourceLine(e.lineno, module)) from e

### TRANSLATION PHASE FOUR: modifying the parse tree

noArgs = ast.arguments(
	args=[], vararg=None,
	kwonlyargs=[], kw_defaults=[],
	kwarg=None, defaults=[])
selfArg = ast.arguments(
	args=[ast.arg(arg='self', annotation=None)], vararg=None,
	kwonlyargs=[], kw_defaults=[],
	kwarg=None, defaults=[])

class AttributeFinder(NodeVisitor):
	"""utility class for finding all referenced attributes of a given name"""
	@staticmethod
	def find(target, node):
		af = AttributeFinder(target)
		af.visit(node)
		return af.attributes

	def __init__(self, target):
		super().__init__()
		self.target = target
		self.attributes = set()

	def visit_Attribute(self, node):
		val = node.value
		if isinstance(val, Name) and val.id == self.target:
			self.attributes.add(node.attr)
		self.visit(val)

class ASTParseError(Exception):
	def __init__(self, line, message):
		super().__init__('Parse error in line ' + str(line) + ': ' + message)

class ASTSurgeon(NodeTransformer):
	def __init__(self, module, constructors):
		super().__init__()
		self.module = module
		self.constructors = { const for const in constructors }

	def parseError(self, node, message):
		line = originalSourceLine(node.lineno, self.module)
		raise ASTParseError(line, message)

	def unpack(self, arg, expected, node):
		"""unpacks arguments to ternary (and up) infix operators"""
		assert expected > 0
		if isinstance(arg, BinOp) and isinstance(arg.op, packageNode):
			if expected == 1:
				raise self.parseError(node, 'gave too many arguments to infix operator')
			else:
				return self.unpack(arg.left, expected - 1, node) + [self.visit(arg.right)]
		elif expected > 1:
			raise self.parseError(node, 'gave too few arguments to infix operator')
		else:
			return [self.visit(arg)]

	def visit_BinOp(self, node):
		left = node.left
		right = node.right
		op = node.op
		if isinstance(op, packageNode):		# unexpected argument package
			raise self.parseError(node, 'unexpected keyword "by"')
		elif type(op) in infixImplementations:	# an operator with non-Python semantics
			arity, implementation = infixImplementations[type(op)]
			assert arity >= 2
			args = [self.visit(left)] + self.unpack(right, arity-1, node)
			newNode = Call(implementation, args, [])
		else:	# all other operators have the Python semantics
			newNode = BinOp(self.visit(left), op, self.visit(right))
		return copy_location(newNode, node)

	def visit_Tuple(self, node):
		if len(node.elts) != 2:
			raise self.parseError(node, 'interval must have exactly two endpoints')
		newElts = [self.visit(elt) for elt in node.elts]
		return copy_location(Call(rangeConstructor, newElts, []), node)

	def visit_Call(self, node):
		func = node.func
		if isinstance(func, Name) and func.id == requireStatement:
			if not (1 <= len(node.args) <= 2):
				raise self.parseError(node, 'require takes exactly one argument')
			if len(node.keywords) != 0:
				raise self.parseError(node, 'require takes no keyword arguments')
			cond = node.args[-1]
			if isinstance(cond, Starred):
				raise self.parseError(node, 'argument unpacking cannot be used with require')
			req = self.visit(cond)
			line = originalSourceLine(node.lineno, self.module)
			newArgs = [Lambda(noArgs, req), Num(line)]
			if len(node.args) == 2:
				prob = node.args[0]
				assert isinstance(prob, Num)
				newArgs.append(prob)
			return copy_location(Call(func, newArgs, []), node)
		else:
			newArgs = []
			for arg in node.args:
				if isinstance(arg, BinOp) and isinstance(arg.op, packageNode):
					newArgs.extend(self.unpack(arg, 2, node))
				else:
					newArgs.append(self.visit(arg))
			newKeywords = [self.visit(kwarg) for kwarg in node.keywords]
			return copy_location(Call(func, newArgs, newKeywords), node)

	def visit_ClassDef(self, node):
		if node.name in self.constructors:		# constructor definition
			newBody = []
			for child in node.body:
				child = self.visit(child)
				if isinstance(child, AnnAssign):	# default value for property
					value = child.annotation
					target = child.target
					metaAttrs = []
					if isinstance(target, Subscript):
						sl = target.slice
						if not isinstance(sl, Index):
							self.parseError(sl, 'malformed attributes for property default')
						sl = sl.value
						if isinstance(sl, Name):
							metaAttrs.append(sl.id)
						elif isinstance(sl, Tuple):
							for elt in sl.elts:
								if not isinstance(elt, Name):
									self.parseError(elt, 'malformed attributes for property default')
								metaAttrs.append(elt.id)
						else:
							self.parseError(sl, 'malformed attributes for property default')
						target = Name(target.value.id, Store())
					properties = AttributeFinder.find('self', value)
					args = [
						Set([Str(prop) for prop in properties]),
						Set([Str(attr) for attr in metaAttrs]),
						Lambda(selfArg, value)
					]
					value = Call(createDefault, args, [])
					newChild = AnnAssign(
						target=target, annotation=value,
						value=None, simple=True)
					child = copy_location(newChild, child)
				newBody.append(child)
			node.body = newBody
			return node
		else:		# ordinary Python class
			# catch some mistakes where 'class' was used instead of 'constructor'
			for base in node.bases:
				name = None
				if isinstance(base, Call):
					name = base.func.id
				elif isinstance(base, Name):
					name = base.id
				if name is not None and name in self.constructors:
					self.parseError(node, 'must use "{}" to subclass objects'.format(constructorStatement))
			return self.generic_visit(node)

def translateParseTree(tree, module, constructors):
	return fix_missing_locations(ASTSurgeon(module, constructors).visit(tree))

### TRANSLATION PHASE FIVE: compilation and execution

class InterpreterParseError(Exception):
	def __init__(self, exc, module, sourceFile):
		tbexc = traceback.TracebackException.from_exception(exc)
		# find line in original code
		found = False
		for frame in reversed(tbexc.stack):
			if frame.filename == sourceFile:
				found = True
				break
		assert found
		line = frame.lineno
		line = line if module is None else originalSourceLine(line, module)
		super().__init__('Parse error in line ' + str(line) + ': ' + tbexc.exc_type.__name__ + ': ' + str(tbexc))

def executeTranslatedTreeIn(tree, namespace, module, filename='<autogenerated Python>'):
	try:
		exec(compile(tree, filename, 'exec'), namespace)
	except Exception as e:
		cause = e if showInternalBacktrace else None
		raise InterpreterParseError(e, module, filename) from cause

### TRANSLATION PHASE SIX: scenario construction

class InvalidScenarioError(Exception):
	pass

def extractScenarioStateFrom(module, namespace, filename):
	internal = namespace[veneerReference]	# get secret reference to veneer state

	# extract created Objects
	module._objects = set(internal.allObjects)
	module._egoObject = internal.egoObject

	# extract global parameters
	module._params = dict(internal.globalParameters)

	# extract requirements and create proper closures
	requirements = internal.allRequirements
	module._requirements = []
	module._requirementDeps = set()		# things needing to be sampled to evaluate the requirements
	def makeClosure(req, bindings, ego, line):
		def closure(values):
			# rebind any names referring to sampled objects
			for name, value in bindings.items():
				if value in values:
					namespace[name] = values[value]
			# rebind ego object, which can be referred to implicitly
			internal.egoObject = values[ego]
			# evaluate requirement condition, reporting errors on the correct line
			try:
				return req()
			except Exception as e:
				cause = e if showInternalBacktrace else None
				raise InterpreterParseError(e, lambda x: line, filename) from cause
		return closure
	for req, (bindings, ego, line, prob) in requirements.items():
		for value in bindings.values():
			if isinstance(value, Samplable):
				module._requirementDeps.add(value)
		if ego is not None:
			assert isinstance(ego, Samplable)
			module._requirementDeps.add(ego)
		module._requirements.append((makeClosure(req, bindings, ego, line), prob))

def constructScenarioFrom(module):
	if module._egoObject is None:
		raise InvalidScenarioError('did not specify ego object')

	params = dict(module._params)
	scenarioTime = params.pop('time', Scenario.defaultTime())
	scenarioWeather = params.pop('weather', Scenario.defaultWeather())
	for param in params:
		print('WARNING: unused scene parameter "{}"'.format(param))

	mappy = module.__dict__['m']		# TODO remove hack
	scenario = Scenario(
		mappy, module._objects, module._egoObject,
		time=scenarioTime, weather=scenarioWeather,
		requirements=module._requirements, requirementDeps=module._requirementDeps)
	return scenario

#####

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('USAGE: python3 translator.py <scenario-module>')
	else:
		# strip extension off filename if necessary
		bits = sys.argv[1].split('.')
		if bits[-1] in ('py',) + dslExtensions:
			del bits[-1]
		name = '.'.join(bits)
		# import module
		startTime = time.time()
		module = importlib.import_module(name)
		# construct scenario
		scenario = constructScenarioFrom(module)
		print('scenario constructed in {:.2f} seconds'.format(time.time() - startTime))
		# generate scenes
		while True:
			scene = scenario.generate()
			for obj in scene.cars:
				obj.position = obj.position.toVector()		# TODO remove hack
				#print(obj.position)
			scene.show(zoom=2)
