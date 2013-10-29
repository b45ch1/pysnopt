#!/usr/bin/env python

import numpy as np
from __future__ import print_function

def userfun(status, x, needF, F, needG, G, cu, iu, ru):
    print('hallo')

def toy0( integer *inform, char *Prob, integer *neF, integer *n, doublereal *ObjAdd,
  integer *ObjRow, doublereal *xlow, doublereal *xupp,
  doublereal *Flow, doublereal *Fupp, doublereal *x,
  integer *xstate, doublereal *Fmul ):
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
  # Give the problem a name.  */

  sprintf(Prob,"%s","Toy0")

  # Assign the dimensions of the constraint Jacobian */

  *neF    = 3
  *n      = 2

  *ObjRow = 1 # NOTE: Me must add one to mesh with fortran */
  *ObjAdd = 0

  # Set the upper and lower bounds. */
  xlow[0]   =   0.0  xlow[1] = -1e6
  xupp[0]   =   1e6  xupp[1] =  1e6
  xstate[0] =   0    xstate[1] = 0

  Flow[0] = -1e6 Flow[1] = -1e6 Flow[2] = -1e6
  Fupp[0] =  1e6 Fupp[1] =  4.0 Fupp[2] =  5.0

  x[0]    = 1.0
  x[1]    = 1.0



int toyusrf_
( integer    *Status, integer *n,    doublereal x[],
  integer    *needF,  integer *neF,  doublereal F[],
  integer    *needG,  integer *neG,  doublereal G[],
  char       *cu,     integer *lencu,
  integer    iu[],    integer *leniu,
  doublereal ru[],    integer *lenru )

  #     ================================================================== */
  #     Computes the nonlinear objective and constraint terms for the toy  */
  #     problem featured in the SnoptA users guide.                        */
  #     neF = 3, n = 2.                                                    */
  #                                                                        */
  #        Minimize                      x(2)                              */
  #                                                                        */
  #        subject to   x(1)**2      + 4 x(2)**2  <= 4,                    */
  #                    (x(1) - 2)**2 +   x(2)**2  <= 5,                    */
  #                     x(1) >= 0.                                         */
  #                                                                        */
  #     ================================================================== */

  F[0] = x[1]
  F[1] = x[0]*x[0] + 4*x[1]*x[1]
  F[2] = (x[0] - 2)*(x[0] - 2) + x[1]*x[1]
  return 0


void toy1
( integer *inform, char *Prob, integer *neF, integer *n,
  integer *iAfun, integer *jAvar, integer *lenA, integer *neA, doublereal *A,
  integer *iGfun, integer *jGvar, integer *lenG, integer *neG, doublereal *ObjAdd,
  integer *ObjRow, doublereal *xlow, doublereal *xupp,
  doublereal *Flow, doublereal *Fupp, doublereal *x,
  integer *xstate, doublereal *Fmul)

  #     ================================================================== */
  #     Toy1   defines input data for the toy problem discussed in the     */
  #     SnoptA Users Guide.                                                */
  #                                                                        */
  #        Minimize                      x(2)                              */
  #                                                                        */
  #        subject to   x(1)**2      + 4 x(2)**2  <= 4,                    */
  #                    (x(1) - 2)**2 +   x(2)**2  <= 5,                    */
  #                     x(1) >= 0.                                         */
  #                                                                        */
  #                                                                        */
  #     On exit:                                                           */
  #        neF  is the number of objective and constraint functions        */
  #               (including linear and nonlinear)                         */
  #        n    is the number of variables.                                */
  #                                                                        */
  #                                                                        */
  #        (iGfun(k),jGvar(k)), k = 1,2,...,neG, define the coordinates    */
  #             of the nonzero problem derivatives.                        */
  #             If (iGfun(k),jGvar(k)) = (i,j), G(k) is the ijth element   */
  #             of the problem vector F(i), i = 0,1,2,...,neF,  with       */
  #             objective function in position 0 and constraint functions  */
  #             in positions  1  through  m.                               */
  #                                                                        */
  #        (iAfun(k),jAvar(k),a(k)), k = 1,2,...,neA, are the coordinates  */
  #             of the nonzero constant problem derivatives.               */
  #                                                                        */
  #             To keep things simple, no constant elements are set here.  */
  #                                                                        */
  #     ================================================================== */
  # Give the problem a name.  */

  strcpy(Prob,"Toy1")

  # Assign the dimensions of the constraint Jacobian */

  *neF    = 3
  *n      = 2

  *ObjRow = 1 # NOTE: Me must add one to mesh with fortran */
  *ObjAdd = 0

  # Set the upper and lower bounds. */
  xlow[0]   =   0.0  xlow[1] = -1e6
  xupp[0]   =   1e6  xupp[1] =  1e6
  xstate[0] =   0    xstate[1] = 0

  Flow[0] = -1e6 Flow[1] = -1e6 Flow[2] = -1e6
  Fupp[0] =  1e6 Fupp[1] =  4.0 Fupp[2] =  5.0
  Fmul[0] =    0 Fmul[1] =    0 Fmul[2] =    0

  x[0]    = 1.0
  x[1]    = 1.0

  *inform = 0
  *neG    = 0
  iGfun[*neG] = 1
  jGvar[*neG] = 1

  *neG    = *neG + 1
  iGfun[*neG] = 1
  jGvar[*neG] = 2

  *neG    = *neG + 1
  iGfun[*neG] = 2
  jGvar[*neG] = 1

  *neG    = *neG + 1
  iGfun[*neG] = 2
  jGvar[*neG] = 2

  *neG    = *neG + 1
  iGfun[*neG] = 3
  jGvar[*neG] = 1

  *neG    = *neG + 1
  iGfun[*neG] = 3
  jGvar[*neG] = 2

  *neG = *neG + 1
  # *neG = 6 */

  *neA = 0



int toyusrfg_
( integer    *Status, integer *n,    doublereal x[],
  integer    *needF,  integer *neF,  doublereal F[],
  integer    *needG,  integer *neG,  doublereal G[],
  char       *cu,     integer *lencu,
  integer    iu[],    integer *leniu,
  doublereal ru[],    integer *lenru )

  #     ==================================================================  */
  #     Computes the nonlinear objective and constraint terms for the toy   */
  #     problem featured in the SnoptA users guide.                         */
  #     neF = 3, n = 2.                                                     */
  #                                                                         */
  #        Minimize                      x(2)                               */
  #                                                                         */
  #        subject to   x(1)**2      + 4 x(2)**2  <= 4,                     */
  #                    (x(1) - 2)**2 +   x(2)**2  <= 5,                     */
  #                     x(1) >= 0.                                          */
  #                                                                         */
  #     The triples (g(k),iGfun(k),jGvar(k)), k = 1,2,...,neG, define       */
  #     the sparsity pattern and values of the nonlinear elements           */
  #     of the Jacobian.                                                    */
  #     ==================================================================  */

  if( *needF > 0 )
    F[0] = x[1] # ! The objective row */
    F[1] = x[0]*x[0] + 4*x[1]*x[1]
    F[2] = (x[0] - 2)*(x[0] - 2) + x[1]*x[1]


  if( *needG > 0 )
    *neG = 0
    # iGfun[*neG] = 1 */
    # jGvar[*neG] = 1 */
    G[*neG] = 0

    # iGfun[*neG] = 1 */
    # jGvar[*neG] = 2 */
    *neG = *neG + 1
    G[*neG] = 1.0

    # iGfun[*neG] = 2 */
    # jGvar[*neG] = 1 */
    *neG = *neG + 1
    G[*neG] = 2*x[0]

    # iGfun[*neG] = 2 */
    # jGvar[*neG] = 2 */
    *neG = *neG + 1
    G[*neG] = 8*x[1]

    # iGfun[*neG] = 3 */
    # jGvar[*neG] = 1 */
    *neG = *neG + 1
    G[*neG] = 2*(x[0] - 2)

    # iGfun[*neG] = 3 */
    # jGvar[*neG] = 2 */
    *neG = *neG + 1
    G[*neG] = 2*x[1]
    *neG = *neG + 1

  return 0

def doublereal(len):
    return np.zeros((length), dtype=np.float64)

def integer(len):
    return np.zeros((length), dtype=np.int64)

def char(len):
    return np.zeros((length), dtype=np.int8)

def main():
  lenrw = 20000
  leniw = 10000
  lencw = 500

  minrw = np.zeros((1), dtype=np.float64)
  miniw = np.zeros((1), dtype=np.int64)
  mincw = np.zeros((1), dtype=np.int8)

  rw = np.zeros((20000,), dtype=np.float64)
  iw = np.zeros((10000,), dtype=np.int64)
  cw = np.zeros((8*500,), dtype=np.int8)

  Cold  = 0
  Basis = 1
  Warm  = 2

  x    = np.zeros((2,), dtype=np.float64)
  xlow = np.zeros((2,), dtype=np.float64)
  xupp = np.zeros((2,), dtype=np.float64)
  xmul = np.zeros((2,), dtype=np.float64)
  F    = np.zeros((3,), dtype=np.float64)
  Flow = np.zeros((3,), dtype=np.float64)
  Fupp = np.zeros((3,), dtype=np.float64)
  Fmul = np.zeros((3,), dtype=np.float64)

  ObjAdd = np.zeros((1,), dtype=np.float64)

  xstate = np.zeros((2,), dtype=np.int64)
  Fstate = np.zeros((3,), dtype=np.int64)

  INFO   = np.zeros((1,), dtype=np.int64)
  ObjRow = np.zeros((1,), dtype=np.int64)
  n      = np.zeros((1,), dtype=np.int64)
  neF    = np.zeros((1,), dtype=np.int64)

  lenA   = np.zeros((1,), dtype=np.int64)
  lenA[0] = 10

  iAfun = np.zeros((10,), dtype=np.int64)
  jAvar = np.zeros((10,), dtype=np.int64)

  A     = np.zeros((1,), dtype=np.float64)

  lenG   = np.zeros((1,), dtype=np.int64)
  lenG[0] = 10

  iGfun = np.zeros((10,), dtype=np.int64)
  jGvar = np.zeros((10,), dtype=np.int64)

  neA = np.zeros((1,), dtype=np.int64)
  neG = np.zeros((1,), dtype=np.int64)

  nxname    = np.zeros((1,), dtype=np.int64)
  nFname    = np.zeros((1,), dtype=np.int64)
  npname = np.zeros((1,), dtype=np.int64)

  nxname[0] = 1
  nFname[0] = 1

  xnames = np.zeros((1*8,), dtype=np.int8)
  Fnames = np.zeros((1*8,), dtype=np.int8)
  Prob   = np.zeros((200*8,), dtype=np.int8)

  iSpecs   = np.zeros((1,), dtype=np.int64)
  spec_len = np.zeros((1,), dtype=np.int64)
  prnt_len = np.zeros((1,), dtype=np.int64)
  iSumm    = np.zeros((1,), dtype=np.int64)
  iPrint   = np.zeros((1,), dtype=np.int64)

  iSpecs[0] = 4
  iSumm [0] = 6
  iPrint[0] = 9

  printname = np.zeros((200*8,), dtype=np.int8)
  specname  = np.zeros((200*8,), dtype=np.int8)

  nS   = np.zeros((1,), dtype=np.int64)
  nInf = np.zeros((1,), dtype=np.int64)

  sInf = np.zeros((1,), dtype=np.float64)

  DerOpt     = np.zeros((1,), dtype=np.int64)
  Major      = np.zeros((1,), dtype=np.int64)
  iSum       = np.zeros((1,), dtype=np.int64)
  iPrt       = np.zeros((1,), dtype=np.int64)
  strOpt_len = np.zeros((1,), dtype=np.int64)
  strOpt     = np.zeros((200*8,), dtype=np.int8)

  print("\nSolving toy0 without first derivatives ...\n")

  # open output files using snfilewrappers.[ch] */
  #sprintf(specname ,   "%s", "sntoya.spc")   spec_len = strlen(specname)
  #sprintf(printname,   "%s", "sntoya.out")   prnt_len = strlen(printname)

  # Open the print file, fortran style */
  #snopenappend_
  #  ( &iPrint, printname,   &INFO, prnt_len )

  #     ================================================================== */
  #     First,  sninit_ MUST be called to initialize optional parameters   */
  #     to their default values.                                           */
  #     ================================================================== */

  sninit_
    ( &iPrint, &iSumm, cw, &lencw, iw, &leniw, rw, &lenrw, 8*500 )

  #     Set up the problem to be solved.                       */
  #     No derivatives are set in this case.                   */
  #     NOTE: To mesh with Fortran style coding,               */
  #           it ObjRow must be treated as if array F          */
  #           started at 1, not 0.  Hence, if F(0) = objective */
  #           then ObjRow should be set to 1.                  */

  toy0
    ( &INFO, Prob, &neF, &n, &ObjAdd, &ObjRow, xlow, xupp,
      Flow, Fupp, x, xstate, Fmul )
  npname = strlen(Prob)

  #     SnoptA will compute the Jacobian by finite-differences.   */
  #     The user has the option of calling  snJac  to define the  */
  #     coordinate arrays (iAfun,jAvar,A) and (iGfun, jGvar).     */

  snjac_
    ( &INFO, &neF, &n, toyusrf_,
      iAfun, jAvar, &lenA, &neA, A,
      iGfun, jGvar, &lenG, &neG,
      x, xlow, xupp, &mincw, &miniw, &minrw,
      cw, &lencw, iw, &leniw, rw, &lenrw,
      cw, &lencw, iw, &leniw, rw, &lenrw,
      8*500, 8*500 )

  #     ------------------------------------------------------------------ */
  #     Warn SnoptA that userf does not compute derivatives.               */
  #     The parameters iPrt and iSum may refer to the Print and Summary    */
  #     file respectively.  Setting them to 0 suppresses printing.         */
  #     ------------------------------------------------------------------ */

  DerOpt = 0
  iPrt   = 0
  iSum   = 0
  sprintf(strOpt,"%s","Derivative option")
  strOpt_len = strlen(strOpt)
  snseti_
    ( strOpt, &DerOpt, &iPrt, &iSum, &INFO,
      cw, &lencw, iw, &leniw, rw, &lenrw, strOpt_len, 8*500 )

  #     ------------------------------------------------------------------ */
  #     Go for it, using a Cold start.                                     */
  #     ------------------------------------------------------------------ */

  snopta_
    ( &Cold, &neF, &n, &nxname, &nFname,
      &ObjAdd, &ObjRow, Prob, toyusrf_,
      iAfun, jAvar, &lenA, &neA, A,
      iGfun, jGvar, &lenG, &neG,
      xlow, xupp, xnames, Flow, Fupp, Fnames,
      x, xstate, xmul, F, Fstate, Fmul,
      &INFO, &mincw, &miniw, &minrw,
      &nS, &nInf, &sInf,
      cw, &lencw, iw, &leniw, rw, &lenrw,
      cw, &lencw, iw, &leniw, rw, &lenrw,
      npname, 8*nxname, 8*nFname,
      8*500, 8*500)


  printf("\nSolving toy1 using first derivatives ...\n")

  toy1
    ( &INFO, Prob, &neF, &n,
      iAfun, jAvar, &lenA, &neA, A,
      iGfun, jGvar, &lenG, &neG,
      &ObjAdd, &ObjRow, xlow, xupp,
      Flow, Fupp, x, xstate, Fmul )

  # Read in specs file (optional) */
  # snfilewrapper_ will open the specs file, fortran style, */
  # then call snspec_ to read in specs.                        */

  snfilewrapper_
    ( specname, &iSpecs, &INFO, cw, &lencw,
      iw, &leniw, rw, &lenrw, spec_len, 8*lencw)

  if( INFO != 101 )

      printf("Warning: trouble reading specs file %s \n", specname)


  # Specify any user options not set in the Specs file. */
  DerOpt = 1
  snseti_
    ( strOpt, &DerOpt, &iPrint, &iSumm, &INFO,
      cw, &lencw, iw, &leniw, rw, &lenrw, strOpt_len, 8*500 )

  Major = 250
  strcpy( strOpt,"Major Iteration limit")
  strOpt_len = strlen(strOpt)
  snseti_
    ( strOpt, &Major, &iPrint, &iSumm, &INFO,
      cw, &lencw, iw, &leniw, rw, &lenrw, strOpt_len, 8*500 )

  #     ------------------------------------------------------------------ */
  #     Solve the problem again, this time with derivatives specified.     */
  #     ------------------------------------------------------------------ */
  snopta_
    ( &Cold, &neF, &n, &nxname, &nFname,
      &ObjAdd, &ObjRow, Prob, toyusrfg_,
      iAfun, jAvar, &lenA, &neA, A,
      iGfun, jGvar, &lenG, &neG,
      xlow, xupp, xnames, Flow, Fupp, Fnames,
      x, xstate, xmul, F, Fstate, Fmul,
      &INFO, &mincw, &miniw, &minrw,
      &nS, &nInf, &sInf,
      cw, &lencw, iw, &leniw, rw, &lenrw,
      cw, &lencw, iw, &leniw, rw, &lenrw,
      npname, 8*nxname, 8*nFname,
      8*500, 8*500)

  snclose_( &iPrint )
  snclose_( &iSpecs )

