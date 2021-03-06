#!/usr/bin/env python
# This file is part of pysnopt, a Python interface to SNOPT.
# Copyright (C) 2013  Manuel Kudruss, Sebastian F. Walter
# License: GPL v3, see LICENSE.txt for details.

import snopt
import numpy as np

def toy0(inform, Prob, neF, n, ObjAdd, ObjRow, xlow, xupp,
         Flow, Fupp, x, xstate, Fmul):

    """
    ==================================================================
    Toy0   defines input data for the toy problem discussed in the
    SnoptA Users Guide.

       Minimize                      x(2)

       subject to   x(1)**2      + 4 x(2)**2  <= 4,
                   (x(1) - 2)**2 +   x(2)**2  <= 5,
                    x(1) >= 0.


    On exit:
    inform      is 0 if there is enough storage, 1 otherwise.
    neF         is the number of problem functions
                (objective and constraints, linear and nonlinear).
    n           is the number of variables.
    xlow        holds the lower bounds on x.
    xupp        holds the upper bounds on x.
    Flow        holds the lower bounds on F.
    Fupp        holds the upper bounds on F.

    xstate(1:n) is a set of initial states for each x  (0,1,2,3,4,5).
    x (1:n)     is a set of initial values for x.
    Fmul(1:neF) is a set of initial values for the dual variables.

    ==================================================================
    """
    # Give the problem a name.
    Prob[:4] = list('Toy0')

    # Assign the dimensions of the constraint Jacobian */

    neF[0]    = 3
    n[0]      = 2

    ObjRow[0] = 1 # NOTE: Me must add one to mesh with fortran */
    ObjAdd[0] = 0

    # Set the upper and lower bounds. */
    xlow[0]   =   0.0
    xlow[1]   =  -1e6
    xupp[0]   =   1e6
    xupp[1]   =   1e6
    xstate[0] =   0
    xstate[1] =   0

    Flow[0] = -1e6
    Flow[1] = -1e6
    Flow[2] = -1e6
    Fupp[0] =  1e6
    Fupp[1] =  4.0
    Fupp[2] =  5.0
    x[0]    =  1.0
    x[1]    =  1.0


def toyusrf(status, x, needF, neF, F, needG, neG, G, cu, iu, ru):

    """
    ==================================================================
    Computes the nonlinear objective and constraint terms for the toy
    problem featured in the SnoptA users guide.
    neF = 3, n = 2.

      Minimize                      x(2)

      subject to   x(1)**2      + 4 x(2)**2  <= 4,
                  (x(1) - 2)**2 +   x(2)**2  <= 5,
                   x(1) >= 0.

    ==================================================================
    """
    # print 'called toyusrf'
    if( needF[0] != 0 ):
        F[0] = x[1]
        F[1] = x[0]*x[0] + 4*x[1]*x[1]
        F[2] = (x[0] - 2)*(x[0] - 2) + x[1]*x[1]
    return 0


def toy1( inform, Prob, neF, n,
          iAfun, jAvar, lenA, neA, A,
          iGfun, jGvar, lenG, neG, ObjAdd,
          ObjRow, xlow, xupp,
          Flow, Fupp, x,
          xstate, Fmul):

    """
    ==================================================================
    Toy1   defines input data for the toy problem discussed in the
    SnoptA Users Guide.

      Minimize                      x(2)

      subject to   x(1)**2      + 4 x(2)**2  <= 4,
                  (x(1) - 2)**2 +   x(2)**2  <= 5,
                   x(1) >= 0.


    On exit:
      neF  is the number of objective and constraint functions
             (including linear and nonlinear)
      n    is the number of variables.


      (iGfun(k),jGvar(k)), k = 1,2,...,neG, define the coordinates
           of the nonzero problem derivatives.
           If (iGfun(k),jGvar(k)) = (i,j), G(k) is the ijth element
           of the problem vector F(i), i = 0,1,2,...,neF,  with
           objective function in position 0 and constraint functions
           in positions  1  through  m.

      (iAfun(k),jAvar(k),a(k)), k = 1,2,...,neA, are the coordinates
           of the nonzero constant problem derivatives.

           To keep things simple, no constant elements are set here.

    ==================================================================
    """


    # Assign the dimensions of the constraint Jacobian

    neF[0]    = 3
    n[0]      = 2

    ObjRow[0] = 1 # NOTE: Me must add one to mesh with fortran
    ObjAdd[0] = 0

    # Set the upper and lower bounds
    xlow[0]   =   0.0
    xlow[1]   = -1e6
    xupp[0]   =   1e6
    xupp[1]   =  1e6
    xstate[0] =   0
    xstate[1] =   0

    Flow[0] = -1e6
    Flow[1] = -1e6
    Flow[2] = -1e6
    Fupp[0] =  1e6
    Fupp[1] =  4.0
    Fupp[2] =  5.0
    Fmul[0] =    0
    Fmul[1] =    0
    Fmul[2] =    0

    x[0]    = 1.0
    x[1]    = 1.0

    inform[0] = 0
    neG[0]    = 0
    iGfun[neG[0]] = 1
    jGvar[neG[0]] = 1

    neG[0]       += 1
    iGfun[neG[0]] = 1
    jGvar[neG[0]] = 2

    neG[0]       += 1
    iGfun[neG[0]] = 2
    jGvar[neG[0]] = 1

    neG[0]       += 1
    iGfun[neG[0]] = 2
    jGvar[neG[0]] = 2

    neG[0]       += 1
    iGfun[neG[0]] = 3
    jGvar[neG[0]] = 1

    neG[0]       += 1
    iGfun[neG[0]] = 3
    jGvar[neG[0]] = 2

    neG[0] += 1
    # neG[0] = 6

    neA[0] = 0


def toyusrfg(status, x, needF, neF, F, needG, neG, G, cu, iu, ru):
    """
    ==================================================================
    Computes the nonlinear objective and constraint terms for the toy
    problem featured in the SnoptA users guide.
    neF = 3, n = 2.
       Minimize                      x(2)
       subject to   x(1)**2      + 4 x(2)**2  <= 4,
                   (x(1) - 2)**2 +   x(2)**2  <= 5,
                    x(1) >= 0.
    The triples (g(k),iGfun(k),jGvar(k)), k = 1,2,...,neG, define
    the sparsity pattern and values of the nonlinear elements
    of the Jacobian.
    ==================================================================
    """

    # print 'called toyusrfg'

    # print 'needF=', needF
    # print 'needG=', needG

    if( needF[0] != 0 ):
        F[0] = x[1] # ! The objective row */
        F[1] = x[0]*x[0] + 4*x[1]*x[1]
        F[2] = (x[0] - 2)*(x[0] - 2) + x[1]*x[1]

    if( needG[0] != 0 ):
        neG[0] = 0
        # iGfun[*neG] = 1 */
        # jGvar[*neG] = 1 */
        G[neG[0]] = 0

        # iGfun[neG] = 1 */
        # jGvar[neG] = 2 */
        neG[0] += 1
        G[neG[0]] = 1.0

        # iGfun[neG] = 2 */
        # jGvar[neG] = 1 */
        neG[0] += 1
        G[neG[0]] = 2*x[0]

        # iGfun[neG] = 2 */
        # jGvar[neG] = 2 */
        neG[0] += 1
        G[neG[0]] = 8*x[1]

        # iGfun[neG] = 3 */
        # jGvar[neG] = 1 */
        neG[0] += 1
        G[neG[0]] = 2*(x[0] - 2)

        # iGfun[neG] = 3 */
        # jGvar[neG] = 2 */
        neG[0] += 1
        G[neG[0]] = 2*x[1]
        neG[0] += 1


def main():
    snopt.check_memory_compatibility()
    minrw = np.zeros((1), dtype=np.int32)
    miniw = np.zeros((1), dtype=np.int32)
    mincw = np.zeros((1), dtype=np.int32)

    rw = np.zeros((10000,), dtype=np.float64)
    iw = np.zeros((10000,), dtype=np.int32)
    cw = np.zeros((8*500,), dtype=np.character)

    Cold  = np.array([0], dtype=np.int32)
    Basis = np.array([1], dtype=np.int32)
    Warm  = np.array([2], dtype=np.int32)

    x    = np.zeros((2,), dtype=np.float64)
    xlow = np.zeros((2,), dtype=np.float64)
    xupp = np.zeros((2,), dtype=np.float64)
    xmul = np.zeros((2,), dtype=np.float64)
    F    = np.zeros((3,), dtype=np.float64)
    Flow = np.zeros((3,), dtype=np.float64)
    Fupp = np.zeros((3,), dtype=np.float64)
    Fmul = np.zeros((3,), dtype=np.float64)

    ObjAdd = np.zeros((1,), dtype=np.float64)

    xstate = np.zeros((2,), dtype=np.int32)
    Fstate = np.zeros((3,), dtype=np.int32)

    INFO   = np.zeros((1,), dtype=np.int32)
    ObjRow = np.zeros((1,), dtype=np.int32)
    n      = np.zeros((1,), dtype=np.int32)
    neF    = np.zeros((1,), dtype=np.int32)

    lenA   = np.zeros((1,), dtype=np.int32)
    lenA[0] = 10

    iAfun = np.zeros((lenA[0],), dtype=np.int32)
    jAvar = np.zeros((lenA[0],), dtype=np.int32)

    A     = np.zeros((lenA[0],), dtype=np.float64)

    lenG   = np.zeros((1,), dtype=np.int32)
    lenG[0] = 10

    iGfun = np.zeros((lenG[0],), dtype=np.int32)
    jGvar = np.zeros((lenG[0],), dtype=np.int32)

    neA = np.zeros((1,), dtype=np.int32)
    neG = np.zeros((1,), dtype=np.int32)

    nxname = np.zeros((1,), dtype=np.int32)
    nFname = np.zeros((1,), dtype=np.int32)

    nxname[0] = 1
    nFname[0] = 1

    xnames = np.zeros((1*8,), dtype=np.character)
    Fnames = np.zeros((1*8,), dtype=np.character)
    Prob   = np.zeros((200*8,), dtype=np.character)

    iSpecs   = np.zeros((1,), dtype=np.int32)
    iSumm    = np.zeros((1,), dtype=np.int32)
    iPrint   = np.zeros((1,), dtype=np.int32)

    iSpecs[0] = 4
    iSumm [0] = 6
    iPrint[0] = 9

    printname = np.zeros((200*8,), dtype=np.character)
    specname  = np.zeros((200*8,), dtype=np.character)

    nS   = np.zeros((1,), dtype=np.int32)
    nInf = np.zeros((1,), dtype=np.int32)

    sInf = np.zeros((1,), dtype=np.float64)

    DerOpt     = np.zeros((1,), dtype=np.int32)
    Major      = np.zeros((1,), dtype=np.int32)
    iSum       = np.zeros((1,), dtype=np.int32)
    iPrt       = np.zeros((1,), dtype=np.int32)
    strOpt     = np.zeros((200*8,), dtype=np.character)

    print("\nSolving toy0 without first derivatives ...\n")

    # open output files using snfilewrappers.[ch] */
    specn  = "sntoya.spc"
    printn = "sntoya.out"
    specname [:len(specn)]  = list(specn)
    printname[:len(printn)] = list(printn)

    # Open the print file, fortran style */
    snopt.snopenappend(iPrint, printname, INFO)

    # ================================================================== */
    # First,  sninit_ MUST be called to initialize optional parameters   */
    # to their default values.                                           */
    # ================================================================== */

    snopt.sninit(iPrint, iSumm, cw, iw, rw)
    # Set up the problem to be solved.                       */
    # No derivatives are set in this case.                   */
    # NOTE: To mesh with Fortran style coding,               */
    #       it ObjRow must be treated as if array F          */
    #       started at 1, not 0.  Hence, if F(0) = objective */
    #       then ObjRow should be set to 1.                  */

    toy0(INFO, Prob, neF, n, ObjAdd, ObjRow, xlow, xupp,
         Flow, Fupp, x, xstate, Fmul)
    # SnoptA will compute the Jacobian by finite-differences.   */
    # The user has the option of calling  snJac  to define the  */
    # coordinate arrays (iAfun,jAvar,A) and (iGfun, jGvar).     */

    snopt.snjac(INFO, neF, n, toyusrf,
                iAfun, jAvar, lenA, neA, A,
                iGfun, jGvar, lenG, neG,
                x, xlow, xupp, mincw, miniw, minrw,
                cw, iw, rw, cw, iw, rw)

    # ------------------------------------------------------------------ */
    # Warn SnoptA that userf does not compute derivatives.               */
    # The parameters iPrt and iSum may refer to the Print and Summary    */
    # file respectively.  Setting them to 0 suppresses printing.         */
    # ------------------------------------------------------------------ */

    DerOpt = np.array([0], dtype=np.int32)
    iPrt   = np.array([0], dtype=np.int32)
    iSum   = np.array([0], dtype=np.int32)
    strOpt_s = "Derivative option"
    strOpt[:len(strOpt_s)] = list(strOpt_s)

    snopt.snseti(strOpt, DerOpt, iPrt, iSum, INFO, cw, iw, rw)
    #     ------------------------------------------------------------------ */
    #     Go for it, using a Cold start.                                     */
    #     ------------------------------------------------------------------ */


    snopt.snopta(Cold, neF, n, nxname, nFname,
           ObjAdd, ObjRow, Prob, toyusrf,
           iAfun, jAvar, lenA, neA, A,
           iGfun, jGvar, lenG, neG,
           xlow, xupp, xnames, Flow, Fupp, Fnames,
           x, xstate, xmul, F, Fstate, Fmul,
           INFO, mincw, miniw, minrw,
           nS, nInf, sInf, cw,  iw,  rw, cw,  iw,  rw )

    print("\nSolving toy1 using first derivatives ...\n")

    toy1(INFO, Prob, neF, n,
         iAfun, jAvar, lenA, neA, A,
         iGfun, jGvar, lenG, neG,
         ObjAdd, ObjRow, xlow, xupp,
         Flow, Fupp, x, xstate, Fmul)

    # Read in specs file (optional)
    # snfilewrapper_ will open the specs file, fortran style,
    # then call snspec_ to read in specs.

    snopt.snfilewrapper(specname, iSpecs, INFO, cw, iw, rw)

    if INFO[0] != 101:
        print("Warning: trouble reading specs file %s \n"%(specname))

    # Specify any user options not set in the Specs file.
    DerOpt[0] = 1
    snopt.snseti(strOpt, DerOpt, iPrint, iSumm, INFO, cw, iw, rw)

    Major = np.array([250], dtype=np.int32)
    strOpt_s = "Major Iteration limit"
    strOpt[:len(strOpt_s)] = list(strOpt_s)
    snopt.snseti( strOpt, Major, iPrint, iSumm, INFO, cw, iw, rw)
    # ------------------------------------------------------------------
    # Solve the problem again, this time with derivatives specified.
    # ------------------------------------------------------------------

    snopt.snopta(Cold, neF, n, nxname, nFname,
           ObjAdd, ObjRow, Prob, toyusrfg,
           iAfun, jAvar, lenA, neA, A,
           iGfun, jGvar, lenG, neG,
           xlow, xupp, xnames, Flow, Fupp, Fnames,
           x, xstate, xmul, F, Fstate, Fmul,
           INFO, mincw, miniw, minrw,
           nS, nInf, sInf, cw, iw, rw, cw, iw, rw)

    snopt.snclose(iPrint)
    snopt.snclose(iSpecs)

if __name__ == '__main__':
    main()

