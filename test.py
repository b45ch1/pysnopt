import snopt
import numpy

iPrint = -1
iSumm  = -1


cw = numpy.array([0]*500*8, dtype=numpy.character)
iw = numpy.array([0]*500,   dtype=numpy.int64)
rw = numpy.array([0]*500,   dtype=numpy.float64)
cu = numpy.array([0]*500*8, dtype=numpy.character)
iu = numpy.array([0]*500,   dtype=numpy.int64)
ru = numpy.array([0]*500,   dtype=numpy.float64)


print cw
print iw
print rw

snopt.check_memory_compatibility()

# snopt.sninit(iPrint, iSumm, cw, iw, rw)

inform = numpy.array([0]*1,   dtype=numpy.int64)
nf     = numpy.array([0]*1,   dtype=numpy.int64)
n      = numpy.array([0]*1,   dtype=numpy.int64)
iafun  = numpy.array([0]*1,   dtype=numpy.int64)
javar  = numpy.array([0]*1,   dtype=numpy.int64)
nea    = numpy.array([0]*1,   dtype=numpy.int64)
lena   = numpy.array([0]*1,   dtype=numpy.int64)

igfun  = numpy.array([0]*1,   dtype=numpy.int64)
jgvar  = numpy.array([0]*1,   dtype=numpy.int64)
neg    = numpy.array([0]*1,   dtype=numpy.int64)
leng   = numpy.array([0]*1,   dtype=numpy.int64)

a      = numpy.array([0]*500,   dtype=numpy.float64)
x      = numpy.array([0]*n[0],   dtype=numpy.float64)
xlow   = numpy.array([0]*n[0],   dtype=numpy.float64)
xupp   = numpy.array([0]*n[0],   dtype=numpy.float64)

mincw  = numpy.array([0]*1,   dtype=numpy.int64)
miniw  = numpy.array([0]*1,   dtype=numpy.int64)
minrw  = numpy.array([0]*1,   dtype=numpy.int64)


def userfun(status, x, needF, F, needG, G, cu, iu, ru):
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