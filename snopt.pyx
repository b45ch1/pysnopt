# author of this file: Sebastian F. Walter, Manuel Kudruss

import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref

cimport snopt

np.import_array()

cdef struct cu_struct:
    char* cu
    void* userfun

cdef int callback(integer    *Status,   integer    *n,
                  doublereal *x,        integer    *needF,
                  integer    *neF,      doublereal *F,
                  integer    *needG,    integer    *neG,
                  doublereal *G,
                  char       *cu,       integer    *lencu,
                  integer    *iu,       integer    *leniu,
                  doublereal *ru,       integer    *lenru):

    cus = <cu_struct*>cu
    cu_p = cus.cu
    userfun_ = cus.userfun

    cdef np.npy_intp shape[1]

    shape[0]  = 1
    status_   = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT64, Status)

    needF_    = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT64, needF)
    needG_    = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT64, needG)

    shape[0]  = leniu[0]
    iu_       = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT64, iu)

    shape[0]  = lencu[0]
    cu_       = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT8, cu_p)

    shape[0]  = n[0]
    x_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, x)
    shape[0]  = neF[0]
    F_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, F)
    shape[0]  = neG[0]
    G_        = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, G)
    shape[0]  = lenru[0]
    ru_       = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, ru)

    (<object>userfun_)(status_, x_, needF_, F_, needG_, G_, cu_, iu_,  ru_)

def check_memory_compatibility():
    assert sizeof(np.int8_t) == sizeof(char), 'sizeof(np.int8_t) != sizeof(char)'
    assert sizeof(np.int64_t) == sizeof(integer), 'sizeof(np.int64_t) != sizeof(integer)'
    assert sizeof(np.float64_t) == sizeof(doublereal), 'sizeof(np.float64_t) != sizeof(doublereal)'

def check_cw_iw_rw(cw, iw, rw):
    assert cw.shape[0] >= 8*500, 'cw.size must be >= 8*500=4000'
    assert iw.shape[0] >= 500, 'iw.size must be >= 500'
    assert rw.shape[0] >= 500, 'rw.size must be >= 500'

def snopta(np.ndarray[np.int8_t,     ndim=1, mode='c'] start,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] nf,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] n,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] nxname,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] nfname,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] objadd,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] objrow,
           np.ndarray[np.int8_t, ndim=1, mode='c']     prob,
           userfg,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] iafun,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] javar,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] lena,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] nea,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] a,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] igfun,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] jgvar,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] leng,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] neg,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] xlow,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] xupp,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] xnames,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] Flow,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] Fupp,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] fnames,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] x,
           np.ndarray[np.int64_t,  ndim=1, mode='c']   xstate,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] xmul,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] f,
           np.ndarray[np.int64_t,  ndim=1, mode='c']   fstate,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] fmul,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] mincw,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] miniw,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] minrw,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] ns,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] ninf,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] sinf,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  cu,
           np.ndarray[np.int64_t,     ndim=1, mode='c'] iu,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  ru,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  cw,
           np.ndarray[np.int64_t,     ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  rw):
    """
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


    cdef cu_struct    cus
    cus.cu = cu.data
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
            <doublereal*> Flow.data,
            <doublereal*> Fupp.data,
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
            lenprob,
            lenxnames,
            lenfnames,
            lencu,
            lencw
            )


def sninit(np.ndarray[np.int64_t,     ndim=1, mode='c'] iPrint,
           np.ndarray[np.int64_t,     ndim=1, mode='c'] iSumm,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,     ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw ):
    """
    """
    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]
    cdef ftnlen str_len = 8

    sninit_( <integer*> iPrint.data,
             <integer*> iSumm.data,
             <char*> cw.data, &lencw,
             <integer*> iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             str_len )

    return None

def sngeti(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] ivalue,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
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


    sngeti_( <char*>       bu.data,
             <integer*>    ivalue.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lenbu, lencw)

def sngetr(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rvalue,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu = bu.shape[0]
    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]

    sngetr_( <char*>       bu.data,
             <doublereal*> rvalue.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lenbu, lencw)

def snset(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
          np.ndarray[np.int64_t,    ndim=1, mode='c'] iprint,
          np.ndarray[np.int64_t,    ndim=1, mode='c'] isumm,
          np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
          np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
          np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
          np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu = bu.shape[0]
    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]

    snset_( <char*>        bu.data,
            <integer*>     iprint.data,
            <integer*>     isumm.data,
            <integer*>     inform.data,
            <char*>        cw.data, &lencw,
            <integer*>     iw.data, &leniw,
            <doublereal*>  rw.data, &lenrw,
            lenbu, lencw)

def sngetc(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] ivalue,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu     = bu.shape[0]
    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]
    cdef integer lenivalue = ivalue.shape[0]

    sngetc_( <char*>       bu.data,
             <char*>       ivalue.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lenbu, lenivalue, lencw)

def snseti(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] ivalue,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iprint,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] isumm,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu     = bu.shape[0]
    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]

    snseti_( <char*>       bu.data,
             <integer*>    ivalue.data,
             <integer*>    iprint.data,
             <integer*>    isumm.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lenbu, lencw)

def snsetr(np.ndarray[np.int8_t,     ndim=1, mode='c'] bu,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rvalue,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iprint,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] isumm,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lenbu     = bu.shape[0]
    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]

    snsetr_( <char*>       bu.data,
             <doublereal*> rvalue.data,
             <integer*>    iprint.data,
             <integer*>    isumm.data,
             <integer*>    inform.data,
             <char*>       cw.data, &lencw,
             <integer*>    iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             lenbu, lencw)

def snspec(np.ndarray[np.int64_t,    ndim=1, mode='c'] ispecs,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
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

def snmema(np.ndarray[np.int64_t,    ndim=1, mode='c'] iexit,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] nf,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] n,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] nxname,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] nfname,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] nea,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] neg,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] mincw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] miniw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] minrw,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
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

def snjac( np.ndarray[np.int64_t,    ndim=1, mode='c'] inform,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] nf,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] n,
           userfg,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iafun,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] javar,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] lena,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] nea,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] a,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] igfun,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] jgvar,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] leng,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] neg,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] x,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] xlow,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] xupp,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] mincw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] miniw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] minrw,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cu,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iu,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] ru,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,    ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw):

    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]
    cdef integer lencu     = cu.shape[0]
    cdef integer leniu     = iu.shape[0]
    cdef integer lenru     = ru.shape[0]

    cdef cu_struct    cus
    cus.cu = cu.data
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

