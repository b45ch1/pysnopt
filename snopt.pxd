## author: Sebastian F. Walter, Manuel Kudruss

cdef extern from "fornorm.h":
    pass

cdef extern from "snopt.h":
    ctypedef void (*snObjectiveFunc) ( int    *mode,
                                       int    *nnObj,
                                       double *x,
                                       double *fObj,
                                       double *gObj,
                                       int    *nState,
                                       char   *cu,
                                       int    *lencu,
                                       double *iu,
                                       int    *leniu,
                                       double *ru,
                                       int    *lenru )

    ctypedef void (*snConstraintFunc) ( int    *mode,
                                        int    *nnCon,
                                        int    *nnJac,
                                        int    *negCon,
                                        double *x,
                                        double *fCon,
                                        double *gCon,
                                        int    *nState,
                                        char   *cu,
                                        int    *lencu,
                                        double *iu,
                                        int    *leniu,
                                        double *ru,
                                        int    *lenru )

    void snopt( char *start, int *m, int *n, int *ne, int *nName,
                int *nnCon, int *nnObj, int *nnJac,
                int *iObj, double *ObjAdd, char Prob[8],
                void (*funcon)( int *mode, int *nnCon, int *nnJac, int *neJac,
                                double *x, double *fCon, double *gCon,
                                int *nState,
                                char cu[][8], int *lencu,
                                int *iu, int *leniu, double *ru, int *lenru,
                                int strlen_cu ),
                void (*funobj)( int *mode, int *nnObj,
                                double *x, double *fObj, double *gObj,
                                int *nState,
                                char cu[][8], int *lencu,
                                int *iu, int *leniu, double *ru, int *lenru,
                                int strlen_cu ),
                double *a, int *ha, int *ka,
                double *bl, double *bu, char Names[][8],
                int *hs, double *xs, double *pi, double *rc,
                int *inform, int *mincw, int *miniw, int *minrw,
                int *nS, int *nInf, double *sInf, double *Obj,
                char cu[][8], int *lencu,
                int *iu, int *leniu, double *ru, int *lenru,
                char cw[][8], int *lencw,
                int *iw, int *leniw, double *rw, int *lenrw,
                int strlen_start, int strlen_Prob, int strlen_Names,
                int strlen_cu, int strlen_cw )

    void sninit( int *iPrint, int *iSumm,
                 char cw[][8], int *lencw,
                 int *iw, int *leniw,
                 double *rw, int *lenrw,
                 int strlen_cw )

    void snspec( int *iSpecs, int *inform,
                 char cw[][8], int *lencw,
                 int *iw, int *leniw,
                 double *rw, int *lenrw,
                 int strlen_cw )

    void snset( char *buffer, int *iPrint,
                int *iSumm, int *inform,
                char cw[][8], int *lencw,
                int *iw, int *leniw,
                double *rw, int *lenrw,
                int strlen_buffer, int strlen_cw )

    void snseti( char *buffer, int *ivalue, int *iPrint,
                 int *iSumm, int *inform,
                 char cw[][8], int *lencw,
                 int *iw, int *leniw,
                 double *rw, int *lenrw,
                 int strlen_buffer, int strlen_cw )

    void snsetr( char *buffer, double *rvalue, int *iPrint, int *iSumm, int *inform,
                 char cw[][8], int *lencw,
                 int *iw, int *leniw,
                 double *rw, int *lenrw,
                 int strlen_buffer, int strlen_cw )

    void sngetc( char *buffer, char *cvalue, int *inform,
                 char cw[][8], int *lencw,
                 int *iw, int *leniw,
                 double *rw, int *lenrw,
                 int strlen_buffer, int strlen_cvalue, int strlen_cw )

    void sngeti( char *buffer, int *ivalue, int *inform,
                 char cw[][8], int *lencw,
                 int *iw, int *leniw,
                 double *rw, int *lenrw,
                 int strlen_buffer, int strlen_cw )

    void sngetr( char *buffer, double *rvalue, int *inform,
                 char cw[][8], int *lencw,
                 int *iw, int *leniw,
                 double *rw, int *lenrw,
                 int strlen_buffer, int strlen_cw )
