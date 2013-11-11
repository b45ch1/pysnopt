# This file is part of pysnopt, a Python interface to SNOPT.
# Copyright (C) 2013  Manuel Kudruss, Sebastian F. Walter
# License: GPL v3, see LICENSE.txt for details.

.SUFFIXES:
.SUFFIXES: .so .o. f

FC        = gfortran
FFLAGS    = -O0 -fPIC -shared -g
LIB       = $(PWD)

FILES     = $(PWD)/../src/sqopt.f \
		    $(PWD)/../src/snopta.f \
		    $(PWD)/../src/snoptb.f \
		    $(PWD)/../src/snoptc.f \
		    $(PWD)/../src/snoptq.f \
		    $(PWD)/../src/npopt.f \
		    $(PWD)/../src/sq02lib.f \
		    $(PWD)/../src/sn02lib.f \
		    $(PWD)/../src/np02lib.f \
		    $(PWD)/../src/sn03prnt.f \
		    $(PWD)/../src/sn05wrpa.f \
		    $(PWD)/../src/sn05wrpb.f \
		    $(PWD)/../src/sn05wrpc.f \
		    $(PWD)/../src/sn05wrpn.f \
		    $(PWD)/../src/sn10mach.f \
		    $(PWD)/../src/sn12ampl.f \
		    $(PWD)/../src/sn17util.f \
		    $(PWD)/../src/sn20amat.f \
		    $(PWD)/../src/sn25bfac.f \
		    $(PWD)/../src/sn27lu.f \
		    $(PWD)/../src/sn30spec.f \
		    $(PWD)/../src/sn35mps.f \
		    $(PWD)/../src/sn37wrap.f \
		    $(PWD)/../src/sn40bfil.f \
		    $(PWD)/../src/sn50lp.f \
		    $(PWD)/../src/sn55qp.f \
		    $(PWD)/../src/sn56qncg.f \
		    $(PWD)/../src/sn57qopt.f \
		    $(PWD)/../src/sn60srch.f \
		    $(PWD)/../src/sn65rmod.f \
		    $(PWD)/../src/sn70nobj.f \
		    $(PWD)/../src/sn80ncon.f \
		    $(PWD)/../src/sn85hess.f \
		    $(PWD)/../src/sn87sopt.f \
		    $(PWD)/../src/sn90lmqn.f \
		    $(PWD)/../src/sn95fmqn.f \
		    $(PWD)/../cppsrc/snfilewrapper.f

all:
	$(FC) -shared -s $(FFLAGS) -o libpysnopt7.so $< $(FILES) -lgfortran -lblas
	python setup.py build_ext --inplace

python:
	python setup.py build_ext --inplace

clean:
	rm -f *.so
	rm -f fort.*
	rm -f *.c
	rm -f *.cpp
	rm -f *.out
