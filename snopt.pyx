# author of this file: Sebastian F. Walter, Manuel Kudruss

import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref

cimport snopt

def check_memory_compatibility():
    assert sizeof(np.int8_t) == sizeof(char), 'sizeof(np.int8_t) != sizeof(char)'
    assert sizeof(np.int64_t) == sizeof(integer), 'sizeof(np.int64_t) != sizeof(integer)'
    assert sizeof(np.float64_t) == sizeof(doublereal), 'sizeof(np.float64_t) != sizeof(doublereal)'

def check_cw_iw_rw(cw, iw, rw):
    assert cw.shape[0] >= 8*500, 'cw.size must be >= 8*500=4000'
    assert iw.shape[0] >= 500, 'iw.size must be >= 500'
    assert rw.shape[0] >= 500, 'rw.size must be >= 500'

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
    check_cw_iw_rw(cu, iu, ru)
    check_cw_iw_rw(cw, iw, rw)

    cdef integer lencu = cu.shape[0]
    cdef integer leniu = iu.shape[0]
    cdef integer lenru = ru.shape[0]

    cdef integer lencw = cw.shape[0]
    cdef integer leniw = iw.shape[0]
    cdef integer lenrw = rw.shape[0]
    cdef ftnlen str_len = 8

    cdef My_fp thisfunc
    #thisfunc = <My_fp> usrfun

    #snopta_(&start,
    #        &nf,
    #        &n,
    #        &nxname,
    #        &nfname,
    #        &objadd,
    #        &objrow,
    #        <char*> prob.data,
    #        thisfunc,
    #        &iafun,
    #        &javar,
    #        &lena,
    #        &nea,
    #        <doublereal*> a.data,
    #        &igfun,
    #        &jgvar,
    #        &leng,
    #        &neg,
    #        <doublereal*> xlow.data,
    #        <doublereal*> xupp.data,
    #        <char*> xnames.data,
    #        <doublereal*> Flow.data,
    #        <doublereal*> Fupp.data,
    #        <char*> fnames.data,
    #        <doublereal*>  x.data,
    #        <integer*>    xstate.data,
    #        <doublereal*> xmul.data,
    #        <doublereal*> f.data,
    #        <integer*>    fstate.data,
    #        <doublereal*> fmul.data,
    #        &inform__,
    #        &mincw,
    #        &miniw,
    #        &minrw,
    #        &ns,
    #        &ninf,
    #        &sinf,
    #        <char*> cu.data,       &lencu,
    #        <integer*> iu.data,    &leniu,
    #        <doublereal*> ru.data, &lenru,
    #        <char*> cw.data,       &lencw,
    #        <integer*> iw.data,    &leniw,
    #        <doublereal*> rw.data, &lenrw,
    #        str_len, # ftnlen prob_len,
    #        str_len, # ftnlen xnames_len,
    #        str_len, # ftnlen fnames_len,
    #        str_len, # ftnlen cu_len,
    #        str_len  # ftnlen cw_len
    #        )

    return None

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

    cdef integer lencu     = cu.shape[0]
    cdef integer lencw     = cw.shape[0]
    cdef integer leniw     = iw.shape[0]
    cdef integer lenrw     = rw.shape[0]

    snjac_( <integer*> inform,
            <integer*> nf,
            <integer*> n,
            userfg,
            <integer*> iafun,
            <integer*> javar,
            <integer*> lena,
            <integer*> nea,
            <doublereal*> a,
            <integer*> igfun,
            <integer*> jgvar,
            <integer*> leng,
            <integer*> neg,
            <doublereal*> x,
            <doublereal*> xlow,
            <doublereal*> xupp,
            <integer*> mincw,
            <integer*> miniw,
            <integer*> minrw,
            <char*> cu,
            <integer*> iu,
            <doublereal*> ru,
            <char*>       cw.data, &lencw,
            <integer*>    iw.data, &leniw,
            <doublereal*> rw.data, &lenrw,
            lencu,
            lencw)

