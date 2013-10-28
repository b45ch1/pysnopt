# author of this file: Sebastian F. Walter

import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref

cimport snopt

def py_sninit(int iPrint,
              int iSumm,
              str cw,
              np.ndarray[int,    ndim=1, mode='c'] iw,
              np.ndarray[double, ndim=1, mode='c'] rw ):
    """
    """
    cw = np.asarray(list[cw], np.str_, 8)

    cdef int cw_shape = cw.shape[1]
    sninit( &iPrint, &iSumm,
            <char(*)[8]> cw.data, <int*> iw.shape.data,
            <int*>       iw.data, <int*> iw.shape.data,
            <double*>    rw.data, <int*> rw.shape.data,
            cw.shape[0] )

    return None

#def py_snspec(np.ndarray[np.double_t, ndim=1] iSpecs,
#              np.ndarray[np.double_t, ndim=1] inform,
#              np.ndarray[np.double_t, ndim=2] cw,
#              np.ndarray[np.double_t, ndim=1] iw,
#              np.ndarray[np.double_t, ndim=1] rw):
#    """
#    """
#    snspec( iSpecs.data, inform.data,
#            cw.data, cw.shape[1],
#            iw.data, iw.shape[0],
#            rw.data, rw.shape[0],
#            cw.shape[0] )
#
#def py_snset( np.ndarray[np.double_t, ndim=1] buf,
#              np.ndarray[np.int, ndim=1] iPrint,
#              np.ndarray[np.int, ndim=1] iSumm,
#              np.ndarray[np.int, ndim=1] inform,
#              np.ndarray[np.str, ndim=2] cw[][8],
#              np.ndarray[np.int, ndim=1] iw,
#              np.ndarray[np.double_t, ndim=1] rw ):
#    """
#    """
#    snset( buf.data, iPrint.data,
#           iSumm.data, inform.data,
#           cw.data, cw.shape[1],
#           iw.data, iw.shape[0],
#           rw.data, rw.shape[0],
#           buf.shape[0],
#           cw.shape[0] )
#
#def py_snseti( np.ndarray[np.str, ndim=2] buf,
#               np.ndarray[np.int, ndim=1] ivalue,
#               np.ndarray[np.int, ndim=1] iPrint,
#               np.ndarray[np.int, ndim=1] iSumm,
#               np.ndarray[np.int, ndim=1] inform,
#               np.ndarray[np.str, ndim=2] cw[][8],
#               np.ndarray[np.int, ndim=1] iw,
#               np.ndarray[np.double_t, ndim=1] rw ):
#    """
#    """
#    snseti( buf.data, ivalue.data, iPrint.data,
#            iSumm.data, inform.data,
#            cw.data, cw.shape[1],
#            iw.data, iw.shape[0],
#            rw.data, rw.shape[0],
#            buf.shape[0], cw.shape[0] )
#
#def py_snsetr( np.ndarray[np.str, ndim=2] buf,
#               np.ndarray[np.int, ndim=1] rvalue,
#               np.ndarray[np.int, ndim=1] iPrint,
#               np.ndarray[np.int, ndim=1] iSumm,
#               np.ndarray[np.int, ndim=1] inform,
#               np.ndarray[np.str, ndim=2] cw[][8],
#               np.ndarray[np.int, ndim=1] iw,
#               np.ndarray[np.double_t, ndim=1] rw ):
#    """
#    """
#    snsetr( buf.data, rvalue.data,
#            iPrint.data, iSumm.data, inform.data,
#            cw.data, cw.shape[0],
#            iw.data, iw.shape[0],
#            rw.data, rw.shape[0],
#            buf.shape[1], cw.shape[1] )
#
#def py_sngetc( np.ndarray[np.str, ndim=2] buf,
#               np.ndarray[np.str, ndim=2] cvalue,
#               np.ndarray[np.int, ndim=1] inform,
#               np.ndarray[np.str, ndim=2] cw,
#               np.ndarray[np.int, ndim=2] iw,
#               np.ndarray[np.double_t, ndim=2] rw ):
#    """
#    """
#    sngetc( buf.data, cvalue.data, inform.data,
#            cw.data, cw.shape[0],
#            iw.data, iw.shape[0],
#            rw.data, rw.shape[0],
#            buf.shape[1], cvalue.shape[0], cw.shape[0] )
#
#def py_sngeti( np.ndrarry[np.str, ndim=2] buf,
#               np.ndrarry[np.int, ndim=1] ivalue,
#               np.ndrarry[np.int, ndim=1] inform,
#               np.ndrarry[np.str, ndim=2] cw[][8],
#               np.ndrarry[np.int, ndim=1] iw,
#               np.ndrarry[np.double_t, ndim=1] rw ):
#    """
#    """
#    sngeti( buf.data, ivalue.data, inform.data,
#            cw.data, cw.shape[0],
#            iw.data, iw.shape[0],
#            rw.data, rw.shape[0],
#            buf.shape[1], cw.shape[1] )
#
#def py_sngetr( np.ndarray[np.str, ndim=2] buf,
#               np.ndarray[np.double_t, ndim=1] rvalue,
#               np.ndarray[np.int, ndim=1] inform,
#               np.ndarray[np.str, ndim=2] cw[][8],
#               np.ndarray[np.int, ndim=2] iw,
#               np.ndarray[np.double_t, ndim=2] rw ):
#    """
#    """
#    sngetr( buf.data, rvalue.data, inform.data,
#            cw.data, cw.shape[1],
#            iw.data, iw.shape[0],
#            rw.data, rw.shape[0],
#            buf.shape[1], cw.shape[1] )

