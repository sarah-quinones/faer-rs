#ifndef CBLAS_H
#define CBLAS_H
#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>
 
 
#ifdef __cplusplus
extern "C" {            /* Assume C declarations for C++ */
#endif /* __cplusplus */
 
/*
 * Enumerated and derived types
 */
#define CBLAS_INDEX size_t /* this may vary between platforms */
 
/*
 * Integer type
 */
#ifndef CBLAS_INT
#ifdef WeirdNEC
   #define CBLAS_INT int64_t
#else
   #define CBLAS_INT int32_t
#endif
#endif
 
/*
 * Integer format string
 */
#ifndef CBLAS_IFMT
#ifdef WeirdNEC
   #define CBLAS_IFMT PRId64
#else
   #define CBLAS_IFMT PRId32
#endif
#endif
 
typedef enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
 
#define CBLAS_ORDER CBLAS_LAYOUT /* this for backward compatibility with CBLAS_ORDER */
 
/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
 
double cblas_dcabs1(const void  *z);
float  cblas_scabs1(const void  *c);
 
float  cblas_sdsdot(const CBLAS_INT N, const float alpha, const float *X,
                    const CBLAS_INT incX, const float *Y, const CBLAS_INT incY);
double cblas_dsdot(const CBLAS_INT N, const float *X, const CBLAS_INT incX, const float *Y,
                   const CBLAS_INT incY);
float  cblas_sdot(const CBLAS_INT N, const float  *X, const CBLAS_INT incX,
                  const float  *Y, const CBLAS_INT incY);
double cblas_ddot(const CBLAS_INT N, const double *X, const CBLAS_INT incX,
                  const double *Y, const CBLAS_INT incY);
 
/*
 * Functions having prefixes Z and C only
 */
void   cblas_cdotu_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotu);
void   cblas_cdotc_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotc);
 
void   cblas_zdotu_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotu);
void   cblas_zdotc_sub(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                       const void *Y, const CBLAS_INT incY, void *dotc);
 
 
/*
 * Functions having prefixes S D SC DZ
 */
float  cblas_snrm2(const CBLAS_INT N, const float *X, const CBLAS_INT incX);
float  cblas_sasum(const CBLAS_INT N, const float *X, const CBLAS_INT incX);
 
double cblas_dnrm2(const CBLAS_INT N, const double *X, const CBLAS_INT incX);
double cblas_dasum(const CBLAS_INT N, const double *X, const CBLAS_INT incX);
 
float  cblas_scnrm2(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
float  cblas_scasum(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
 
double cblas_dznrm2(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
double cblas_dzasum(const CBLAS_INT N, const void *X, const CBLAS_INT incX);
 
 
/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX cblas_isamax(const CBLAS_INT N, const float  *X, const CBLAS_INT incX);
CBLAS_INDEX cblas_idamax(const CBLAS_INT N, const double *X, const CBLAS_INT incX);
CBLAS_INDEX cblas_icamax(const CBLAS_INT N, const void   *X, const CBLAS_INT incX);
CBLAS_INDEX cblas_izamax(const CBLAS_INT N, const void   *X, const CBLAS_INT incX);
 
/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */
 
/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void cblas_sswap(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                 float *Y, const CBLAS_INT incY);
void cblas_scopy(const CBLAS_INT N, const float *X, const CBLAS_INT incX,
                 float *Y, const CBLAS_INT incY);
void cblas_saxpy(const CBLAS_INT N, const float alpha, const float *X,
                 const CBLAS_INT incX, float *Y, const CBLAS_INT incY);
 
void cblas_dswap(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                 double *Y, const CBLAS_INT incY);
void cblas_dcopy(const CBLAS_INT N, const double *X, const CBLAS_INT incX,
                 double *Y, const CBLAS_INT incY);
void cblas_daxpy(const CBLAS_INT N, const double alpha, const double *X,
                 const CBLAS_INT incX, double *Y, const CBLAS_INT incY);
 
void cblas_cswap(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
void cblas_ccopy(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
void cblas_caxpy(const CBLAS_INT N, const void *alpha, const void *X,
                 const CBLAS_INT incX, void *Y, const CBLAS_INT incY);
 
void cblas_zswap(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
void cblas_zcopy(const CBLAS_INT N, const void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY);
void cblas_zaxpy(const CBLAS_INT N, const void *alpha, const void *X,
                 const CBLAS_INT incX, void *Y, const CBLAS_INT incY);
 
 
/*
 * Routines with S and D prefix only
 */
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void cblas_srotm(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                 float *Y, const CBLAS_INT incY, const float *P);
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void cblas_drotm(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                 double *Y, const CBLAS_INT incY, const double *P);
 
 
 
/*
 * Routines with S D C Z CS and ZD prefixes
 */
void cblas_sscal(const CBLAS_INT N, const float alpha, float *X, const CBLAS_INT incX);
void cblas_dscal(const CBLAS_INT N, const double alpha, double *X, const CBLAS_INT incX);
void cblas_cscal(const CBLAS_INT N, const void *alpha, void *X, const CBLAS_INT incX);
void cblas_zscal(const CBLAS_INT N, const void *alpha, void *X, const CBLAS_INT incX);
void cblas_csscal(const CBLAS_INT N, const float alpha, void *X, const CBLAS_INT incX);
void cblas_zdscal(const CBLAS_INT N, const double alpha, void *X, const CBLAS_INT incX);
 
void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_drotg(double *a, double *b, double *c, double *s);
void cblas_crotg(void *a, void *b, float *c, void *s);
void cblas_zrotg(void *a, void *b, double *c, void *s);
 
void cblas_srot(const CBLAS_INT N, float *X, const CBLAS_INT incX,
                float *Y, const CBLAS_INT incY, const float c, const float s);
void cblas_drot(const CBLAS_INT N, double *X, const CBLAS_INT incX,
                double *Y, const CBLAS_INT incY, const double c, const double  s);
void cblas_csrot(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY, const float c, const float s);
void cblas_zdrot(const CBLAS_INT N, void *X, const CBLAS_INT incX,
                 void *Y, const CBLAS_INT incY, const double c, const double s);
 
/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */
 
/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemv(const CBLAS_LAYOUT layout,
                 const CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float *X, const CBLAS_INT incX, const float beta,
                 float *Y, const CBLAS_INT incY);
void cblas_sgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const float alpha,
                 const float *A, const CBLAS_INT lda, const float *X,
                 const CBLAS_INT incX, const float beta, float *Y, const CBLAS_INT incY);
void cblas_strmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *A, const CBLAS_INT lda,
                 float *X, const CBLAS_INT incX);
void cblas_stbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const float *A, const CBLAS_INT lda,
                 float *X, const CBLAS_INT incX);
void cblas_stpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *Ap, float *X, const CBLAS_INT incX);
void cblas_strsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *A, const CBLAS_INT lda, float *X,
                 const CBLAS_INT incX);
void cblas_stbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const float *A, const CBLAS_INT lda,
                 float *X, const CBLAS_INT incX);
void cblas_stpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const float *Ap, float *X, const CBLAS_INT incX);
 
void cblas_dgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double *X, const CBLAS_INT incX, const double beta,
                 double *Y, const CBLAS_INT incY);
void cblas_dgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const double alpha,
                 const double *A, const CBLAS_INT lda, const double *X,
                 const CBLAS_INT incX, const double beta, double *Y, const CBLAS_INT incY);
void cblas_dtrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *A, const CBLAS_INT lda,
                 double *X, const CBLAS_INT incX);
void cblas_dtbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const double *A, const CBLAS_INT lda,
                 double *X, const CBLAS_INT incX);
void cblas_dtpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *Ap, double *X, const CBLAS_INT incX);
void cblas_dtrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *A, const CBLAS_INT lda, double *X,
                 const CBLAS_INT incX);
void cblas_dtbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const double *A, const CBLAS_INT lda,
                 double *X, const CBLAS_INT incX);
void cblas_dtpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const double *Ap, double *X, const CBLAS_INT incX);
 
void cblas_cgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta,
                 void *Y, const CBLAS_INT incY);
void cblas_cgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y, const CBLAS_INT incY);
void cblas_ctrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
void cblas_ctbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
void cblas_ctpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);
void cblas_ctrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda, void *X,
                 const CBLAS_INT incX);
void cblas_ctbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
void cblas_ctpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);
 
void cblas_zgemv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *X, const CBLAS_INT incX, const void *beta,
                 void *Y, const CBLAS_INT incY);
void cblas_zgbmv(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE TransA, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT KL, const CBLAS_INT KU, const void *alpha,
                 const void *A, const CBLAS_INT lda, const void *X,
                 const CBLAS_INT incX, const void *beta, void *Y, const CBLAS_INT incY);
void cblas_ztrmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
void cblas_ztbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
void cblas_ztpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);
void cblas_ztrsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *A, const CBLAS_INT lda, void *X,
                 const CBLAS_INT incX);
void cblas_ztbsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const CBLAS_INT K, const void *A, const CBLAS_INT lda,
                 void *X, const CBLAS_INT incX);
void cblas_ztpsv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                 const CBLAS_INT N, const void *Ap, void *X, const CBLAS_INT incX);
 
 
/*
 * Routines with S and D prefixes only
 */
void cblas_ssymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *X, const CBLAS_INT incX,
                 const float beta, float *Y, const CBLAS_INT incY);
void cblas_ssbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *X, const CBLAS_INT incX,
                 const float beta, float *Y, const CBLAS_INT incY);
void cblas_sspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const float alpha, const float *Ap,
                 const float *X, const CBLAS_INT incX,
                 const float beta, float *Y, const CBLAS_INT incY);
void cblas_sger(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                const float alpha, const float *X, const CBLAS_INT incX,
                const float *Y, const CBLAS_INT incY, float *A, const CBLAS_INT lda);
void cblas_ssyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, float *A, const CBLAS_INT lda);
void cblas_sspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, float *Ap);
void cblas_ssyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, const float *Y, const CBLAS_INT incY, float *A,
                const CBLAS_INT lda);
void cblas_sspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const float *X,
                const CBLAS_INT incX, const float *Y, const CBLAS_INT incY, float *A);
 
void cblas_dsymv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *X, const CBLAS_INT incX,
                 const double beta, double *Y, const CBLAS_INT incY);
void cblas_dsbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *X, const CBLAS_INT incX,
                 const double beta, double *Y, const CBLAS_INT incY);
void cblas_dspmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const double alpha, const double *Ap,
                 const double *X, const CBLAS_INT incX,
                 const double beta, double *Y, const CBLAS_INT incY);
void cblas_dger(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                const double alpha, const double *X, const CBLAS_INT incX,
                const double *Y, const CBLAS_INT incY, double *A, const CBLAS_INT lda);
void cblas_dsyr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, double *A, const CBLAS_INT lda);
void cblas_dspr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, double *Ap);
void cblas_dsyr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, const double *Y, const CBLAS_INT incY, double *A,
                const CBLAS_INT lda);
void cblas_dspr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const double *X,
                const CBLAS_INT incX, const double *Y, const CBLAS_INT incY, double *A);
 
 
/*
 * Routines with C and Z prefixes only
 */
void cblas_chemv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
void cblas_chbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
void cblas_chpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *Ap,
                 const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
void cblas_cgeru(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
void cblas_cgerc(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
void cblas_cher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const void *X, const CBLAS_INT incX,
                void *A, const CBLAS_INT lda);
void cblas_chpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const float alpha, const void *X,
                const CBLAS_INT incX, void *A);
void cblas_cher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
void cblas_chpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *Ap);
 
void cblas_zhemv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
void cblas_zhbmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
void cblas_zhpmv(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 const CBLAS_INT N, const void *alpha, const void *Ap,
                 const void *X, const CBLAS_INT incX,
                 const void *beta, void *Y, const CBLAS_INT incY);
void cblas_zgeru(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
void cblas_zgerc(CBLAS_LAYOUT layout, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *X, const CBLAS_INT incX,
                 const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
void cblas_zher(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const void *X, const CBLAS_INT incX,
                void *A, const CBLAS_INT lda);
void cblas_zhpr(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                const CBLAS_INT N, const double alpha, const void *X,
                const CBLAS_INT incX, void *A);
void cblas_zher2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *A, const CBLAS_INT lda);
void cblas_zhpr2(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo, const CBLAS_INT N,
                const void *alpha, const void *X, const CBLAS_INT incX,
                const void *Y, const CBLAS_INT incY, void *Ap);
 
/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
 
/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const float alpha, const float *A,
                 const CBLAS_INT lda, const float *B, const CBLAS_INT ldb,
                 const float beta, float *C, const CBLAS_INT ldc);
void cblas_ssymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float *B, const CBLAS_INT ldb, const float beta,
                 float *C, const CBLAS_INT ldc);
void cblas_ssyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 const float beta, float *C, const CBLAS_INT ldc);
void cblas_ssyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const float alpha, const float *A, const CBLAS_INT lda,
                  const float *B, const CBLAS_INT ldb, const float beta,
                  float *C, const CBLAS_INT ldc);
void cblas_strmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 float *B, const CBLAS_INT ldb);
void cblas_strsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const float alpha, const float *A, const CBLAS_INT lda,
                 float *B, const CBLAS_INT ldb);
 
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const double alpha, const double *A,
                 const CBLAS_INT lda, const double *B, const CBLAS_INT ldb,
                 const double beta, double *C, const CBLAS_INT ldc);
void cblas_dsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double *B, const CBLAS_INT ldb, const double beta,
                 double *C, const CBLAS_INT ldc);
void cblas_dsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 const double beta, double *C, const CBLAS_INT ldc);
void cblas_dsyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const double alpha, const double *A, const CBLAS_INT lda,
                  const double *B, const CBLAS_INT ldb, const double beta,
                  double *C, const CBLAS_INT ldc);
void cblas_dtrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 double *B, const CBLAS_INT ldb);
void cblas_dtrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const double alpha, const double *A, const CBLAS_INT lda,
                 double *B, const CBLAS_INT ldb);
 
void cblas_cgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc);
void cblas_csymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
void cblas_csyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *beta, void *C, const CBLAS_INT ldc);
void cblas_csyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const void *beta,
                  void *C, const CBLAS_INT ldc);
void cblas_ctrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
void cblas_ctrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
 
void cblas_zgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INT M, const CBLAS_INT N,
                 const CBLAS_INT K, const void *alpha, const void *A,
                 const CBLAS_INT lda, const void *B, const CBLAS_INT ldb,
                 const void *beta, void *C, const CBLAS_INT ldc);
void cblas_zsymm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
void cblas_zsyrk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *beta, void *C, const CBLAS_INT ldc);
void cblas_zsyr2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const void *beta,
                  void *C, const CBLAS_INT ldc);
void cblas_ztrmm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
void cblas_ztrsm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 void *B, const CBLAS_INT ldb);
 
 
/*
 * Routines with prefixes C and Z only
 */
void cblas_chemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
void cblas_cherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const float alpha, const void *A, const CBLAS_INT lda,
                 const float beta, void *C, const CBLAS_INT ldc);
void cblas_cher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const float beta,
                  void *C, const CBLAS_INT ldc);
 
void cblas_zhemm(CBLAS_LAYOUT layout, CBLAS_SIDE Side,
                 CBLAS_UPLO Uplo, const CBLAS_INT M, const CBLAS_INT N,
                 const void *alpha, const void *A, const CBLAS_INT lda,
                 const void *B, const CBLAS_INT ldb, const void *beta,
                 void *C, const CBLAS_INT ldc);
void cblas_zherk(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                 CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                 const double alpha, const void *A, const CBLAS_INT lda,
                 const double beta, void *C, const CBLAS_INT ldc);
void cblas_zher2k(CBLAS_LAYOUT layout, CBLAS_UPLO Uplo,
                  CBLAS_TRANSPOSE Trans, const CBLAS_INT N, const CBLAS_INT K,
                  const void *alpha, const void *A, const CBLAS_INT lda,
                  const void *B, const CBLAS_INT ldb, const double beta,
                  void *C, const CBLAS_INT ldc);
 
void cblas_xerbla(CBLAS_INT p, const char *rout, const char *form, ...);
 
#ifdef __cplusplus
}
#endif
#endif