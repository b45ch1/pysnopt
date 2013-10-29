# author: Sebastian F. Walter, Manuel Kudruss

cdef extern from "stdio.h":
    void * memcpy ( void * destination, const void * source, size_t num )

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

cdef extern from "snfilewrapper.hh":
  void snopenappend_(integer *iunit, char *name,
                     integer *inform, ftnlen name_len)

  void snfilewrapper_(char *name__, integer *ispec, integer *inform__,
                      char *cw, integer *lencw, integer *iw,
                      integer *leniw, doublereal *rw, integer *lenrw,
                      ftnlen name_len, ftnlen cw_len)

  void snclose_(integer *iunit)

  void snopenread_(integer *iunit, char *name, integer *inform,
              ftnlen name_len)

