# author of this file: Sebastian F. Walter

import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref

cimport snopt

cdef public void dummy(integer *Status, integer *n,
        doublereal x[],     integer *needF,
        integer *neF,  doublereal F[],
        integer    *needG,  integer *neG,  doublereal G[],
        char       *cu,     integer *lencu,
        integer    iu[],    integer *leniu,
        doublereal ru[],    integer *lenru ):
    pass


def snopta(integer start,
           integer nf,
           integer n,
           integer nxname,
           integer nfname,
           doublereal objadd,
           integer objrow,
           np.ndarray[np.int8_t, ndim=1, mode='c'] prob,
           usrfun,
           integer iafun,
           integer javar,
           integer lena,
           integer nea,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] a,
           integer igfun,
           integer jgvar,
           integer leng,
           integer neg,
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
           integer inform__,
           integer mincw,
           integer miniw,
           integer minrw,
           integer ns,
           integer ninf,
           doublereal sinf,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  cu,
           np.ndarray[np.int64_t,     ndim=1, mode='c'] iu,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  ru,
           np.ndarray[np.int8_t,     ndim=1, mode='c']  cw,
           np.ndarray[np.int64_t,     ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c']  rw,
           ftnlen prob_len,
           ftnlen xnames_len,
           ftnlen fnames_len,
           ftnlen cu_len,
           ftnlen cw_len):
    """
    """
    assert cu.shape[0] >= 8*500, 'cw.size must be >= 8*500=4000'
    assert iu.shape[0] >= 500, 'iw.size must be >= 500'
    assert ru.shape[0] >= 500, 'rw.size must be >= 500'

    assert cw.shape[0] >= 8*500, 'cw.size must be >= 8*500=4000'
    assert iw.shape[0] >= 500, 'iw.size must be >= 500'
    assert rw.shape[0] >= 500, 'rw.size must be >= 500'

    cdef integer lencu = cu.shape[0]
    cdef integer leniu = iu.shape[0]
    cdef integer lenru = ru.shape[0]

    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]
    cdef ftnlen str_len = 8

    snopta_(&start,
            &nf,
            &n,
            &nxname,
            &nfname,
            &objadd,
            &objrow,
            <char*> prob.data,
            dummy,
            &iafun,
            &javar,
            &lena,
            &nea,
            <doublereal*> a.data,
            &igfun,
            &jgvar,
            &leng,
            &neg,
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
            &inform__,
            &mincw,
            &miniw,
            &minrw,
            &ns,
            &ninf,
            &sinf,
            <char*> cu.data,       &lencu,
            <integer*> iu.data,    &leniu,
            <doublereal*> ru.data, &lenru,
            <char*> cw.data,       &lencw,
            <integer*> iw.data,    &leniw,
            <doublereal*> rw.data, &lenrw,
            str_len, # ftnlen prob_len,
            str_len, # ftnlen xnames_len,
            str_len, # ftnlen fnames_len,
            str_len, # ftnlen cu_len,
            str_len  # ftnlen cw_len
            )

    return None


def sninit(integer iPrint,
           integer iSumm,
           np.ndarray[np.int8_t,     ndim=1, mode='c'] cw,
           np.ndarray[np.int64_t,     ndim=1, mode='c'] iw,
           np.ndarray[np.float64_t,  ndim=1, mode='c'] rw ):
    """
    """

    assert cw.shape[0] >= 8*500, 'cw.size must be >= 8*500=4000'
    assert iw.shape[0] >= 500, 'iw.size must be >= 500'
    assert rw.shape[0] >= 500, 'rw.size must be >= 500'

    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]
    cdef ftnlen str_len = 8

    sninit_( &iPrint, &iSumm,
             <char*> cw.data, &lencw,
             <integer*> iw.data, &leniw,
             <doublereal*> rw.data, &lenrw,
             str_len )

    return None
