# This file is part of pysnopt, a Python interface to SNOPT.
# Copyright (C) 2013  Manuel Kudruss, Sebastian F. Walter
# License: GPL v3, see LICENSE.txt for details.

import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref

cimport snopt

np.import_array()


cdef struct cu_struct:
    # ===============================================================
    # A hack to pass userfun through SNOPT via cu.
    # Note that if SNOPT requires len(cu) >= len(cu_struct.cu)
    # you have to increase len(cu_struct.cu) here and recompile.
    # ===============================================================
    char  cu[16000]
    void* userfun

cdef int callback(integer    *Status,   integer    *n,
                  doublereal *x,        integer    *needF,
                  integer    *neF,      doublereal *F,
                  integer    *needG,    integer    *neG,
                  doublereal *G,
                  char       *cu,       integer    *lencu,
                  integer    *iu,       integer    *leniu,
                  doublereal *ru,       integer    *lenru):
    # ===============================================================
    # C callback function that is passed as argument to the fortran
    # subroutine snopta_.

    # The pointer cu is casted to cu_struct, which contains a pointer
    # to the Python callable.
    # ===============================================================

    # print 'called callback'
    # print 'needF', needF[0]
    # print 'needG', needG[0]
    # print 'neF', neF[0]
    # print 'neG', neG[0]
    # printf("neG=%ld\n", neG[0])

    cus = <cu_struct*>cu
    cdef np.npy_intp shape[1]

    shape[0]  = 1
    status_   = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, Status)
    needF_    = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, needF)

    if needF[0] > 0:
        neF_      = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, neF)
        shape[0]  = neF[0]
        F_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, F)
    else:
        shape[0]  = 0
        neF_      = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, neF)
        shape[0]  = 0
        F_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, F)

    shape[0]  = 1
    needG_    = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, needG)

    if needG[0] > 0:
        shape[0]  = 1
        neG_      = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, neG)
        shape[0]  = neG[0]
        G_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, G)
    else:
        shape[0]  = 0
        neG_      = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, neG)
        shape[0]  = 0
        G_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, G)

    shape[0]  = leniu[0]
    iu_       = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT32, iu)

    shape[0]  = lencu[0]
    cu_       = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT8, cus.cu)

    shape[0]  = n[0]
    x_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, x)

    shape[0]  = lenru[0]
    ru_       = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, ru)

    (<object>cus.userfun)(status_, x_, needF_, neF_, F_, needG_, neG_, G_, cu_, iu_,  ru_)
    return 0

def check_memory_compatibility():
    """
    Check that the number of bytes of a char, integer and doublereal
    match the assumptions.

    If this check fails, you have to adapt the type casts in ``callback``, etc.
    """
    assert sizeof(np.int8_t) == sizeof(char), 'sizeof(np.int8_t) != sizeof(char)'
    assert sizeof(np.int32_t) == sizeof(integer), 'sizeof(np.int32_t) != sizeof(integer)'
    assert sizeof(np.float64_t) == sizeof(doublereal), 'sizeof(np.float64_t) != sizeof(doublereal)'

def check_cw_iw_rw(cw, iw, rw):
    assert cw.shape[0] >= 8*500, 'cw.size must be >= 8*500=4000'
    assert iw.shape[0] >= 500, 'iw.size must be >= 500'
    assert rw.shape[0] >= 500, 'rw.size must be >= 500'

def snopta(np.ndarray[np.int32_t,    ndim=1, mode='c']  start,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  nf,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  n,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  nxname,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  nfname,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  objadd,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  objrow,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  prob,
           userfg,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  iafun,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  javar,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  lena,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  nea,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  a,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  igfun,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  jgvar,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  leng,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  neg,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  xlow,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  xupp,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  xnames,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  flow,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  fupp,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  fnames,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  x,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  xstate,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  xmul,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  f,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  fstate,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  fmul,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  inform,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  mincw,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  miniw,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  minrw,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  ns,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  ninf,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  sinf,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  cu,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  iu,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  ru,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c']  iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  rw):
    """
*     ==================================================================
*     snOptA  is a Fortran wrappper for the SNOPT solver.
*     snOptA   is a subroutine for constrained nonlinear
*     optimization.  The optimization problem involves m  functions
*     F(1), F(2), ... , F(nF), each a function of n variables
*     x(1), x(2), ... , x(n).  The  problem has the form:
*
*           minimize/maximize    ObjAdd + F(ObjRow)
*
*                            ( xlow <=  x  <= xupp,
*                 subject to (
*                            ( Flow <=  F  <= Fupp,
*
*     where ObjAdd is a constant, ObjRow is a user-specified row of  F,
*     xlow, Flow, xupp and Fupp are constant lower and upper bounds.
*
*     ------------------------------------------------------------------
*     NOTE: Before calling SNOPTA, your calling program MUST call the
*     initialization routine using the call:
*     call snInit( iPrint, iSumm, cw, lencw, iw, leniw, rw, lenrw )
*     This sets the default values of the optional parameters. You can
*     also alter the default values of iPrint and iSumm before snOptA
*     is used.  iPrint = 0, etc, is OK.
*     ------------------------------------------------------------------
*
*     o If ObjRow = 0, then snOptA will find a point satisfying the
*       constraints.
*
*     o If all functions are linear, F = A x for some sparse matrix A.
*       This defines a linear program (LP).  In this case,  the nonzero
*       elements of A can be input in coordinate form (i,j,A_ij) (see
*       below).
*
*     o If all functions are nonlinear, F = F(x) for some vector
*       F(x) of smooth functions.  In this case, the elements of  F  and
*       (optionally) their first and second partial derivatives must be
*       coded by the user in the subroutine usrfun  (see below).
*
*     o If some functions are linear and some are nonlinear, the user
*       can choose to set every component in usrfun.  It is usually more
*       efficient, however,  to supply the coefficients of the linear
*       functions via the sparse array  A (see below).   In this case,
*       the linear elements of  F  need not be assigned (SNOPTA will
*       figure out which components of  F  are needed).
*
*     o In the most general situation, the ith component of F(x) is the
*       sum of linear and nonlinear terms.  In this case, if F(x) can be
*       defined as a sum of "non-overlapping" linear and nonlinear
*       functions, then the nonlinear part of F can be defined in usrfun
*       and the linear part can be defined via the array A.
*
*       Suppose that the ith component of F(x) is of the form
*            F_i(x) = f_i(x) + sum (over j)  A_ij x_j,
*       where f_i(x) is a nonlinear function and the elements A_ij
*       are constant.   It is convenient to write  F_i(x)  in the
*       compact form  F_i(x) = f_i(x) + A_i' x, where A_i denotes a
*       column vector with components ( A_i1, A_i2, ..., A_in ), and
*       "A_i'" denotes the transpose of A_i.
*
*       Functions f_i and A_i are said to be "non-overlapping" if any
*       variable x_j  appearing explicitly in f_i(x) does not appear
*       explicitly in A_i'x, i.e., A_ij = 0.  (Equivalently, any
*       variable with a nonzero A_ij must not appear explicitly in
*       f_i(x).)  For example, the function
*         F_i(x) = 3x_1 + exp(x_2)x_4 + x_2^2 + 4x_4 - x_3 + x_5
*       can be written as the sum of non-overlapping functions f_i and
*       A_i'x, such that
*           f_i(x) = exp(x_2)x_4 + x_2^2  + 4x_4  and
*           A_i'x  = 3x_1 - x_3 + x_5.
*
*       Given a non-overlapping sum for each component of F, we can
*       write  F(x) = f(x) + Ax, where f(x) is a vector-valued function
*       of x and A is a sparse matrix whose ith row is A_i'.
*
*       The nF by n  Jacobian of  F(x)  is the sum of two  nF by n
*       sparse matrices G and A,  i.e.,  J = G + A,  where G and A
*       contain the nonlinear and constant elements of J respectively.
*       The important property of non-overlapping functions is that
*       a nonzero entry of J is either an element of A, or an element
*       of G, but NOT BOTH (i.e., the nonzeros of  A  and  G  do not
*       overlap.
*
*       The nonzero elements of A and G must be provided in coordinate
*       form.  In coordinate form, a nonzero element G_ij of a matrix
*       G  is stored as the triple (i,j,G_ij).  The kth coordinate is
*       defined by iGfun(k) and jGvar(k)  (i.e., if i=iGfun(k) and
*       j=jGvar(k), then G(k) is the ijth element of G.)  Any known
*       values of G(k) must be assigned by the user in the routine
*       usrfun.
*
*       RESTRICTIONS:
*        1.  If the elements of G cannot be provided because they are
*            either too expensive or too complicated to evaluate,  it
*            is still necessary to specify the position of the nonzeros
*            as specified by the arrays iGfun and jGvar.
*
*        2.  If an element of G happens to be zero at a given point,
*            it must still be loaded in usrfun. (The order of the
*            list of coordinates (triples) is meaningful in snOptA.)
*
*       The elements of A and G can be stored in any order, (e.g., by
*       rows, by columns, or mixed). Duplicate entries are ignored.
*
*     ON ENTRY:
*
*     Start   specifies how a starting basis (and certain other items)
*             are to be obtained.
*             start =  0 (Cold) means that Crash should be used to
*                      choose an initial basis, unless a basis file is
*                      given by reference in the Specs file to an
*                      Old basis file.
*             start =  1 (Basis file) means the same (but is more
*                      meaningful in the latter case).
*             start =  2 (Warm) means that a basis is already defined
*                      in xstate and Fstate (probably from an earlier
*                      call).
*
*     nF      is the number  of problem functions in F, including the
*             objective function (if any) and the linear
*             and nonlinear constraints.  Simple upper and lower bound
*             constraints on the variables should not be included in  F.
*             nF > 0.
*
*     n       is the number of variables.
*             n > 0.
*
*     neA     is the number of nonzero entries in A.
*             neA >= 0.
*
*     nxname  is the number of 8-character column (i.e., variable) names
*             provided in the array xnames.  If nxname = 1,  then there
*             are NO column names (generic names will be used in the
*             printed solution).  Otherwise, nxname = n and every
*             column name must be provided.
*
*     nFname  is the number of 8-character row (i.e., constraint and
*             objective) names provided in the array Fnames.
*             If nFname = 1,  then there are NO row names (generic
*             names will be used in the printed solution).  Otherwise,
*             nFname = nF and every row name must be provided.
*
*     ObjAdd  is a constant that will be added to the objective.
*             Typically ObjAdd = 0.0d+0.
*
*     Prob    is an 8-character name for the problem, used in the
*             output.  A blank name can be assigned if necessary.
*
*     xlow(n) are the lower bounds on x.
*
*     xupp(n) are the upper bounds on x.
*
*     xnames(nxname) is an character*8 array of names for each x(j).
*             If nxname =  1, xnames is not used.  The printed solution
*             will use generic names for the variables.
*             If nxname = n, xnames(j) should contain an 8 character
*             name of the jth variable (j = 1:n).
*
*     Flow(n) are the lower bounds on F.  If component F(ObjRow)
*             is being optimized,  Flow(ObjRow) is ignored.
*
*     Fupp(n) are the upper bounds on F.  If component F(ObjRow)
*             is being optimized,  Fupp(ObjRow) is ignored.
*
*     Fnames(nFname) is an character*8 array of names for each F(i).
*             If nFname =  1, Fnames is not used.  The printed solution
*             will use generic names for the objective and constraints.
*             If nName = nF, Fnames(j) should contain an 8 character
*             name of the jth constraint (j=1:nF).
*
*     xstate(n) sometimes contains a set of initial states for each
*             variable x.  See the following NOTES.
*
*     x(n)    is a set of initial values for each variable  x.
*
*  NOTES:  1. If start = 0 (Cold) or 1 (Basis file) and an OLD BASIS
*             file is to be input, xstate and x need not be set at all.
*
*          2. Otherwise, xstate(1:n) must be defined for a cold start.
*             If nothing special is known about the problem, or if
*             there is no wish to provide special information,
*             you may set xstate(j) = 0, x(j) = 0.0d+0 for all j=1:n.
*             All variables will be eligible for the initial basis.
*
*             Less trivially, to say that variable j will probably
*             be equal to one of its bounds,
*             set xstate(j) = 4 and x(j) = bl(j)
*             or  xstate(j) = 5 and x(j) = bu(j) as appropriate.
*
*          3. For Cold starts with no basis file, a Crash procedure
*             is used to select an initial basis.  The initial basis
*             matrix will be triangular (ignoring certain small
*             entries in each column).
*             The values xstate(j) = 0, 1, 2, 3, 4, 5 have the following
*             meaning:
*
*             xstate(j)  State of variable j during Crash
*             ---------  --------------------------------
*             0, 1, 3    Eligible for the basis.  3 is given preference.
*             2, 4, 5    Ignored.
*
*             After Crash, xstate(j) = 2 entries are made superbasic.
*             Other entries not selected for the basis are made
*             nonbasic at the value x(j) if bl(j) <= x(j) <= bu(j),
*             or at the value bl(j) or bu(j) closest to x(j).
*
*          4. For Warm starts, all of Fstate(1:nF) is assumed to be
*             set to the values 0, 1, 2 or 3 from some previous call.
*
*     Fmul(nF) contains an estimate of the Lagrange multipliers
*             (shadow prices) for the F- constraints.  They are used
*             to define the Lagrangian for the first major iteration.
*             If nothing is known about Fmul, set
*             Fmul(i) = 0.0d+0, i = 1:nF
*
*     ON EXIT:
*
*     xstate(n) is the final state vector for x:
*
*                hs(j)    State of variable j    Normal value of x(j)
*
*                  0      nonbasic               bl(j)
*                  1      nonbasic               bu(j)
*                  2      superbasic             Between bl(j) and bu(j)
*                  3      basic                  ditto
*
*             Very occasionally there may be nonbasic variables for
*             which x(j) lies strictly between its bounds.
*             If nInf = 0, basic and superbasic variables may be outside
*             their bounds by as much as the Feasibility tolerance.
*             Note that if Scale is specified, the Feasibility tolerance
*             applies to the variables of the SCALED problem.
*             In this case, the variables of the original problem may be
*             as much as 0.1 outside their bounds, but this is unlikely
*             unless the problem is very badly scaled.
*
*     x(n)    contains the final variables.
*
*     F(nF)   contains the final values of F.
*
*     xmul(nF) is the vector of Lagrange multipliers (shadow prices)
*             for the variables constraints.
*
*     Fmul(nF) is the vector of Lagrange multipliers (shadow prices)
*             for the general constraints.
*
*     INFO    says what happened; see the User's Guide.
*             The possible values are as follows:
*
*             INFO       Meaning
*
*                0    finished successfully
*                1       optimality conditions satisfied
*                2       feasible point found
*                3       requested accuracy could not be achieved
*
*               10    the problem appears to be infeasible
*               11       infeasible linear constraints
*               12       infeasible linear equalities
*               13       nonlinear infeasibilities minimized
*               14       infeasibilities minimized
*
*               20    the problem appears to be unbounded
*               21       unbounded objective
*               22       constraint violation limit reached
*
*               30    resource limit error
*               31       iteration limit reached
*               32       major iteration limit reached
*               33       the superbasics limit is too small
*
*               40    terminated after numerical difficulties
*               41       current point cannot be improved
*               42       singular basis
*               43       cannot satisfy the general constraints
*               44       ill-conditioned null-space basis
*
*               50    error in the user-supplied functions
*               51       incorrect objective  derivatives
*               52       incorrect constraint derivatives
*
*               60    undefined user-supplied functions
*               61       undefined function at the first feasible point
*               62       undefined function at the initial point
*               63       unable to proceed into undefined region
*
*               70    user requested termination
*               71       terminated during function evaluation
*               72       terminated during constraint evaluation
*               73       terminated during objective evaluation
*               74       terminated from monitor routine
*
*               80    insufficient storage allocated
*               81       work arrays must have at least 500 elements
*               82       not enough character storage
*               83       not enough integer storage
*               84       not enough real storage
*
*               90    input arguments out of range
*               91       invalid input argument
*               92       basis file dimensions do not match this problem
*               93       the QP Hessian is indefinite
*
*              140    system error
*              141       wrong no of basic variables
*              142       error in basis package
*
*     mincw   says how much character storage is needed to solve the
*             problem.  If INFO = 82, the work array cw(lencw) was
*             too small.  snOptA may be called again with lencw suitably
*             larger than mincw.
*
*     miniw   says how much integer storage is needed to solve the
*             problem.  If INFO = 83, the work array iw(leniw) was too
*             small.  snOptA may be called again with leniw suitably
*             larger than miniw.  (The bigger the better, since it is
*             not certain how much storage the basis factors need.)
*
*     minrw   says how much real storage is needed to solve the
*             problem.  If INFO = 84, the work array rw(lenrw) was too
*             small.  (See the comments above for miniw.)
*
*     nS      is the final number of superbasics.
*
*     nInf    is the number of infeasibilities.
*
*     sInf    is the sum    of infeasibilities.
*
*     cu(lencu), iu(leniu), ru(lenru)  are character, integer and real
*             arrays of USER workspace.  These arrays are available to
*             pass data to the user-defined routine usrfun.
*             If no workspace is required, you can either use dummy
*             arrays for cu, iu and ru, or use cw, iw and rw
*             (see below).
*
*     cw(lencw), iw(leniw), rw(lenrw)  are character*8, integer and real
*             arrays of workspace used by snOptA.
*             lencw should be at least 500, or nF+n if names are given.
*                              +.
*             leniw should be about max( 500, 20(nF+n) ) or larger.
*             lenrw should be about max( 500, 40(nF+n) ) or larger.
*
*     SNOPT package maintained by Philip E. Gill,
*     Dept of Mathematics, University of California, San Diego.
*
*     ==================================================================
    """
    check_cw_iw_rw(cu, iu, ru)
    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencu = cu.shape[0]
    cdef integer leniu = iu.shape[0]
    cdef integer lenru = ru.shape[0]

    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]


    cdef integer lenprob   = prob.shape[0]
    cdef integer lenxnames = xnames.shape[0]
    cdef integer lenfnames = fnames.shape[0]

    cdef integer prob_len   = strlen(prob.data)

    # input checks
    assert nf[0] == f.shape[0],      'error: nf[0] != f.shape[0]'
    assert nf[0] == flow.shape[0],   'error: nf[0] != flow.shape[0]'
    assert nf[0] == fupp.shape[0],   'error: nf[0] != fupp.shape[0]'
    assert nf[0] == fmul.shape[0],   'error: nf[0] != fmul.shape[0]'
    assert nf[0] == fstate.shape[0], 'error: nf[0] != fstate.shape[0]'

    assert n[0] == x.shape[0],       'error: n[0] != x.shape[0]'
    assert n[0] == xlow.shape[0],    'error: n[0] != xlow.shape[0]'
    assert n[0] == xupp.shape[0],    'error: n[0] != xupp.shape[0]'
    assert n[0] == xstate.shape[0],  'error: n[0] != xstate.shape[0]'

    assert 0 <= nea[0],              'error: nea < 0'
    assert 1 <= lena[0],             'error: lena < 1'
    assert nea <= lena[0],           'error: nea > lena'

    assert 0 <= neg[0],              'error: neg < 0'
    assert 1 <= leng[0],             'error: leng < 1'
    assert neg <= leng[0],           'error: neg > leng'


    cdef cu_struct cus
    assert sizeof(cus.cu) >= <size_t>lencu, '%d >= %d, please change length cu_struct.cu and recompile the interface'%(sizeof(cus.cu), <size_t>lencu)
    memcpy(cus.cu, cu.data, lencu*sizeof(char));
    cus.userfun = <void*> userfg

    snopta_(<integer*> start.data,
            <integer*> nf.data,
            <integer*> n.data,
            <integer*> nxname.data,
            <integer*> nfname.data,
            <doublereal*> objadd.data,
            <integer*> objrow.data,
            <char*> prob.data,
            callback,
            <integer*> iafun.data,
            <integer*> javar.data,
            <integer*> lena.data,
            <integer*> nea.data,
            <doublereal*> a.data,
            <integer*> igfun.data,
            <integer*> jgvar.data,
            <integer*> leng.data,
            <integer*> neg.data,
            <doublereal*> xlow.data,
            <doublereal*> xupp.data,
            <char*> xnames.data,
            <doublereal*> flow.data,
            <doublereal*> fupp.data,
            <char*> fnames.data,
            <doublereal*>  x.data,
            <integer*>    xstate.data,
            <doublereal*> xmul.data,
            <doublereal*> f.data,
            <integer*>    fstate.data,
            <doublereal*> fmul.data,
            <integer*> inform.data,
            <integer*> mincw.data,
            <integer*> miniw.data,
            <integer*> minrw.data,
            <integer*> ns.data,
            <integer*> ninf.data,
            <doublereal*> sinf.data,
            <char*> &cus,          &lencu,
            <integer*> iu.data,    &leniu,
            <doublereal*> ru.data, &lenru,
            <char*> cw.data,       &lencw,
            <integer*> iw.data,    &leniw,
            <doublereal*> rw.data, &lenrw,
            prob_len,
            lenxnames,
            lenfnames,
            lencu,
            lencw
            )

def sninit(np.ndarray[np.int32_t,     ndim=1, mode='c'] iPrint,
           np.ndarray[np.int32_t,     ndim=1, mode='c'] iSumm,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,     ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw ):
    """
    """
    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]

    sninit_( <integer*> iPrint.data,
             <integer*> iSumm.data,
             <char*> cw.data, &lencw,
             <integer*> iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lencw )

def sngeti(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] ivalue,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    """

    Remark
    ------

    compare to ./cwrap/snopt.c:389
    and     to ./cppsrc/snoptProblem.cc:332
    """

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu = bu.shape[0]
    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]
    cdef integer bu_len   = strlen(bu.data)

    sngeti_( <char*>       bu.data,
             <integer*>    ivalue.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             bu_len, lencw)

def sngetr(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rvalue,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu = bu.shape[0]
    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]
    cdef integer bu_len   = strlen(bu.data)

    sngetr_( <char*>       bu.data,
             <doublereal*> rvalue.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             bu_len, lencw)

def snset(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
          np.ndarray[np.int32_t,    ndim=1, mode='c'] iprint,
          np.ndarray[np.int32_t,    ndim=1, mode='c'] isumm,
          np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
          np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
          np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
          np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu = bu.shape[0]
    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]
    cdef integer bu_len   = strlen(bu.data)

    snset_( <char*>        bu.data,
            <integer*>     iprint.data,
            <integer*>     isumm.data,
            <integer*>     inform.data,
            <char*>        cw.data, &lencw,
            <integer*>     iw.data, &leniw,
            <doublereal*>  rw.data, &lenrw,
            bu_len, lencw)

def sngetc(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cvalue,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu       = bu.shape[0]
    cdef integer lencw       = cw.shape[0]
    cdef integer leniw       = iw.shape[0]
    cdef integer lenrw       = rw.shape[0]
    cdef integer lencvalue   = cvalue.shape[0]
    cdef integer bu_len      = strlen(bu.data)
    cdef integer cvalue_len  = strlen(cvalue.data)

    sngetc_( <char*>       bu.data,
             <char*>       cvalue.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             bu_len, cvalue_len, lencw)

def snseti(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] ivalue,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iprint,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] isumm,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu     = bu.shape[0]
    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]
    cdef integer bu_len    = strlen(bu.data)

    snseti_( <char*>       bu.data,
             <integer*>    ivalue.data,
             <integer*>    iprint.data,
             <integer*>    isumm.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             bu_len, lencw)

def snsetr(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rvalue,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iprint,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] isumm,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu     = bu.shape[0]
    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]
    cdef integer bu_len    = strlen(bu.data)

    snsetr_( <char*>       bu.data,
             <doublereal*> rvalue.data,
             <integer*>    iprint.data,
             <integer*>    isumm.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             bu_len, lencw)

def snspec(np.ndarray[np.int32_t,    ndim=1, mode='c'] ispecs,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]

    snspec_( <integer*>    ispecs.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lencw)

def snmema(np.ndarray[np.int32_t,    ndim=1, mode='c'] iexit,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] nf,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] n,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] nxname,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] nfname,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] nea,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] neg,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] mincw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] miniw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] minrw,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]

    snmema_( <integer*>    iexit.data,
             <integer*>    nf.data,
             <integer*>    n.data,
             <integer*>    nxname.data,
             <integer*>    nfname.data,
             <integer*>    nea.data,
             <integer*>    neg.data,
             <integer*>    mincw.data,
             <integer*>    miniw.data,
             <integer*>    minrw.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lencw)

def snjac( np.ndarray[np.int32_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] nf,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] n,
           userfg,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iafun,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] javar,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] lena,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] nea,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] a,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] igfun,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] jgvar,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] leng,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] neg,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] x,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] xlow,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] xupp,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] mincw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] miniw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] minrw,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cu,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iu,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] ru,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int32_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]
    cdef integer lencu     = cu.shape[0]
    cdef integer leniu     = iu.shape[0]
    cdef integer lenru     = ru.shape[0]

    cdef cu_struct    cus
    assert sizeof(cus.cu) >= <size_t>lencu, '%d >= %d, please change length cu_struct.cu and recompile the interface'%(sizeof(cus.cu), <size_t>lencu)
    memcpy(cus.cu, cu.data, lencu*sizeof(char));
    cus.userfun = <void*> userfg

    snjac_( <integer*>    inform.data,
            <integer*>    nf.data,
            <integer*>    n.data,
            callback,
            <integer*>    iafun.data,
            <integer*>    javar.data,
            <integer*>    lena.data,
            <integer*>    nea.data,
            <doublereal*> a.data,
            <integer*>    igfun.data,
            <integer*>    jgvar.data,
            <integer*>    leng.data,
            <integer*>    neg.data,
            <doublereal*> x.data,
            <doublereal*> xlow.data,
            <doublereal*> xupp.data,
            <integer*>    mincw.data,
            <integer*>    miniw.data,
            <integer*>    minrw.data,
            <char*>       &cus, &lencu,
            <integer*>    iu.data, &leniu,
            <doublereal*> ru.data, &lenru,
            <char*>       cw.data, &lencw,
            <integer*>    iw.data, &leniw,
            <doublereal*> rw.data, &lenrw,
            lencu,
            lencw)

def snopenappend(np.ndarray[np.int32_t, ndim=1, mode='c'] iunit,
                 np.ndarray[np.int8_t,  ndim=1, mode='c'] name,
                 np.ndarray[np.int32_t, ndim=1, mode='c'] inform):
    """
    """

    cdef integer name_len = strlen(name.data)

    snopenappend_(<integer*> iunit.data,
                  <char*> name.data,
                  <integer*> inform.data,
                  name_len)

def snfilewrapper(np.ndarray[np.int8_t,    ndim=1, mode='c'] name,
                  np.ndarray[np.int32_t,   ndim=1, mode='c'] ispec,
                  np.ndarray[np.int32_t,   ndim=1, mode='c'] inform,
                  np.ndarray[np.int8_t,    ndim=1, mode='c'] cw,
                  np.ndarray[np.int32_t,   ndim=1, mode='c'] iw,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] rw):
    """
    """
    cdef integer lencw    = cw.shape[0]
    cdef integer leniw    = iw.shape[0]
    cdef integer lenrw    = rw.shape[0]
    cdef integer name_len = strlen(name.data)


    snfilewrapper_(<char*> name.data,
                   <integer*> ispec.data,
                   <integer*> inform.data,
                   <char*> cw.data,       &lencw,
                   <integer*> iw.data,    &leniw,
                   <doublereal*> rw.data, &lenrw,
                   name_len,
                   lencw)

def snclose(np.ndarray[np.int32_t, ndim=1, mode='c'] iunit):
    """
    """
    snclose_(<integer*> iunit.data)

def snopenread(np.ndarray[np.int32_t, ndim=1, mode='c'] iunit,
               np.ndarray[np.int8_t,  ndim=1, mode='c'] name,
               np.ndarray[np.int32_t, ndim=1, mode='c'] inform):
    """
    """

    cdef integer name_len = strlen(name.data)

    snopenread_(<integer*> iunit.data,
                <char*> name.data,
                <integer*> inform.data,
                name_len)

