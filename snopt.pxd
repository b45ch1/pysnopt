## author: Sebastian F. Walter, Manuel Kudruss

cdef extern from "f2c.h":
    ctypedef int ftnlen
    ctypedef long int integer
    ctypedef unsigned long int uinteger
    ctypedef char *address
    ctypedef short int shortint
    ctypedef float real
    ctypedef double doublereal
    ctypedef long int logical
    ctypedef short int shortlogical
    ctypedef char logical1
    ctypedef char integer1

cdef extern from "snopt.hh":
    void sninit_( integer *iPrint, integer *iSumm, char *cw,
       integer *lencw, integer *iw, integer *leniw,
       doublereal *rw, integer *lenrw, ftnlen cw_len )

    void sngeti_( char *buffer, integer *ivalue, integer *inform__,
       char *cw, integer *lencw, integer *iw,
       integer *leniw, doublereal *rw, integer *lenrw,
       ftnlen buffer_len, ftnlen cw_len)

    void sngetr_( char *buffer, doublereal *ivalue, integer *inform__,
           char *cw, integer *lencw, integer *iw,
           integer *leniw, doublereal *rw, integer *lenrw,
           ftnlen buffer_len, ftnlen cw_len)

    void snset_( char *buffer, integer *iprint, integer *isumm,
       integer *inform__, char *cw, integer *lencw,
       integer *iw, integer *leniw,
       doublereal *rw, integer *lenrw,
       ftnlen buffer_len, ftnlen cw_len)