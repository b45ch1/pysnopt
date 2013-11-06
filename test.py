# This file is part of pysnopt, a Python interface to SNOPT.
# Copyright (C) 2013  Manuel Kudruss, Sebastian F. Walter
# License: GPL v3, see LICENSE.txt for details.

import snopt
import numpy

iPrint = -1
iSumm  = -1


cw = numpy.array([0]*500*8, dtype=numpy.character)
iw = numpy.array([0]*500,   dtype=numpy.int32)
rw = numpy.array([0]*500,   dtype=numpy.float64)
cu = numpy.array([0]*500*8, dtype=numpy.character)
iu = numpy.array([0]*500,   dtype=numpy.int32)
ru = numpy.array([0]*500,   dtype=numpy.float64)


print cw
print iw
print rw

snopt.check_memory_compatibility()

# snopt.sninit(iPrint, iSumm, cw, iw, rw)

inform = numpy.array([0]*1,   dtype=numpy.int32)
nf     = numpy.array([0]*1,   dtype=numpy.int32)
n      = numpy.array([0]*1,   dtype=numpy.int32)
iafun  = numpy.array([0]*1,   dtype=numpy.int32)
javar  = numpy.array([0]*1,   dtype=numpy.int32)
nea    = numpy.array([0]*1,   dtype=numpy.int32)
lena   = numpy.array([0]*1,   dtype=numpy.int32)

igfun  = numpy.array([0]*1,   dtype=numpy.int32)
jgvar  = numpy.array([0]*1,   dtype=numpy.int32)
neg    = numpy.array([0]*1,   dtype=numpy.int32)
leng   = numpy.array([0]*1,   dtype=numpy.int32)

a      = numpy.array([0]*500,   dtype=numpy.float64)
x      = numpy.array([0]*n[0],   dtype=numpy.float64)
xlow   = numpy.array([0]*n[0],   dtype=numpy.float64)
xupp   = numpy.array([0]*n[0],   dtype=numpy.float64)

mincw  = numpy.array([0]*1,   dtype=numpy.int32)
miniw  = numpy.array([0]*1,   dtype=numpy.int32)
minrw  = numpy.array([0]*1,   dtype=numpy.int32)


def userfun(status, x, needF, neF, F, needG, neG, G, cu, iu, ru):
      print 'hallo'

snopt.snjac( inform,
             nf,
             n,
             userfun,
             iafun,
             javar,
             lena,
             nea,
             #
             a,
             #
             igfun,
             jgvar,
             leng,
             neg,
             #
             x,
             xlow,
             xupp,
             mincw,
             miniw,
             minrw,
             cu,
             iu,
             ru,
             cw,
             iw,
             rw)


print 'done'