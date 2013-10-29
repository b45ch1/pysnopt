import snopt
import numpy

iPrint = -1
iSumm  = -1
cw = numpy.array([0]*500*8, dtype=numpy.character)
iw = numpy.array([0]*500,   dtype=numpy.int64)
rw = numpy.array([0]*500,   dtype=numpy.float64)


print cw
print iw
print rw

snopt.check_memory_compatibility()

# snopt.sninit(iPrint, iSumm, cw, iw, rw)

print 'done'