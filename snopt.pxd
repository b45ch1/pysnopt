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

    void sngetc_( char *buffer, char *ivalue, integer *inform__,
       char *cw, integer *lencw, integer *iw,
       integer *leniw, doublereal *rw, integer *lenrw,
       ftnlen buffer_len, ftnlen ivalue_len, ftnlen cw_len)

    void snseti_( char *buffer, integer *ivalue, integer *iprint,
       integer *isumm, integer *inform__, char *cw,
       integer *lencw, integer *iw, integer *leniw,
       doublereal *rw, integer *lenrw, ftnlen buffer_len,
       ftnlen cw_len)

    void snsetr_( char *buffer, doublereal *rvalue, integer * iprint,
       integer *isumm, integer *inform__, char *cw,
       integer *lencw, integer *iw, integer *leniw,
       doublereal *rw, integer *lenrw, ftnlen buffer_len,
       ftnlen cw_len)

    void snspec_( integer *ispecs, integer *inform__, char *cw,
         integer *lencw, integer *iw, integer *leniw,
         doublereal *rw, integer *lenrw, ftnlen cw_len)

    void snmema_( integer *iexit, integer *nf, integer *n, integer *nxname,
       integer *nfname, integer *nea, integer *neg,
       integer *mincw, integer *miniw,
       integer *minrw, char *cw, integer *lencw, integer *iw,
       integer *leniw, doublereal *rw, integer *lenrw,
       ftnlen cw_len)

    void snjac_( integer *inform__, integer *nf, integer *n, My_fp userfg,
         integer *iafun, integer *javar, integer *lena,
         integer *nea, doublereal *a, integer *igfun,
         integer *jgvar, integer *leng, integer *neg,
         doublereal *x, doublereal *xlow, doublereal *xupp,
         integer *mincw, integer *miniw,
         integer *minrw, char *cu, integer *lencu,
         integer *iu, integer *leniu, doublereal *ru,
         integer *lenru, char *cw, integer *lencw, integer *iw,
         integer *leniw, doublereal *rw, integer *lenrw,
         ftnlen cu_len, ftnlen cw_len )
