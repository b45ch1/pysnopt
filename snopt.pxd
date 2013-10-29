# author: Sebastian F. Walter, Manuel Kudruss

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
    ctypedef int (*My_fp)( integer *Status, integer *n,
            doublereal x[],     integer *needF,
            integer *neF,  doublereal F[],
            integer    *needG,  integer *neG,  doublereal G[],
            char       *cu,     integer *lencu,
            integer    iu[],    integer *leniu,
            doublereal ru[],    integer *lenru )

    void snopta_( integer *start, integer *nf, integer *n,
            integer *nxname, integer *nfname, doublereal *objadd, integer *objrow,
            char *prob, My_fp usrfun, integer *iafun, integer *javar,
            integer *lena, integer *nea, doublereal *a, integer *igfun,
            integer *jgvar, integer *leng, integer *neg, doublereal *xlow,
            doublereal *xupp, char *xnames, doublereal *flow, doublereal *fupp,
            char *fnames, doublereal *x, integer *xstate, doublereal *xmul,
            doublereal *f, integer *fstate, doublereal *fmul, integer *inform__,
            integer *mincw, integer *miniw, integer *minrw, integer *ns,
            integer *ninf, doublereal *sinf, char *cu, integer *lencu, integer *iu,
            integer *leniu, doublereal *ru, integer *lenru, char *cw, integer *lencw,
            integer *iw, integer *leniw, doublereal *rw, integer *lenrw,
            ftnlen prob_len, ftnlen xnames_len, ftnlen fnames_len, ftnlen cu_len,
            ftnlen cw_len)

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

