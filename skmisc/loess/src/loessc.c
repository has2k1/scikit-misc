/*
 * The authors of this software are Cleveland, Grosse, and Shyu.
 * Copyright (c) 1989, 1992 by AT&T.
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHORS NOR AT&T MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 */
#include "S.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* Declarations */

static void
loess_workspace(int *d, int *n, double *span, int *degree,
                int *nonparametric, int *drop_square,
                int *sum_drop_sqr, int *setLf);

static void
loess_prune(int *parameter, int *a, double *xi, double *vert,
            double *vval);

static void
loess_grow(int *parameter, int *a, double *xi, double *vert,
           double *vval);

static void
loess_free(void);

#define MIN(x,y)  ((x) < (y) ? (x) : (y))
#define MAX(x,y)  ((x) > (y) ? (x) : (y))
#define GAUSSIAN  1
#define SYMMETRIC 0

static int *iv = NULL, liv, lv, tau;
static double *v = NULL;

extern char *error_message;
extern int error_status;


void F77_SUB(lowesa)(double*, int*, int*, int*, int*, double*, double*);
void F77_SUB(lowesb)(double*, double*, double*, double*, int*, int*, double*);
void F77_SUB(lowesc)(int*, double*, double*, double*, double*, double*);
void F77_SUB(lowesd)(int*, int*, int*, double*, int*, int*,
                     double*, int*, int*, int*, int*);
void F77_SUB(lowese)(int*, double*, int*, double*, double*);
void F77_SUB(lowesf)(double*, double*, double*, int*, double*,
                     int*, double*, double*, int*, double*);
void F77_SUB(lowesl)(int*, double*, int*, double*, double*);
void F77_SUB(ehg169)(int*, int*, int*, int*, int*, int*,
                     double*, int*, double*, int*, int*, int*);
void F77_SUB(ehg196)(int*, int*, double*, double*);
void F77_SUB(ehg182)(int *i);
void F77_SUB(ehg183a)(char *s, int *nc,int *i,int *n,int *inc);
void F77_SUB(ehg184a)(char *s, int *nc, double *x, int *n, int *inc);


void
loess_raw(double *y, double *x, double *weights, double *robust, int *d,
          int *n, double *span, int *degree, int *nonparametric,
          int *drop_square, int *sum_drop_sqr, double *cell, char **surf_stat,
          double *surface, int *parameter, int *a, double *xi, double *vert,
          double *vval, double *diagonal, double *trL, double *one_delta,
          double *two_delta, int *setLf)
{
    int zero = 0, one = 1, two = 2, nsing, i, k;
    double *hat_matrix, *LL, dzero = 0;


    *trL = 0;
    loess_workspace(d, n, span, degree, nonparametric, drop_square,
                    sum_drop_sqr, setLf);
    v[1] = *cell;
    if(!strcmp(*surf_stat, "interpolate/none")) {
        F77_SUB(lowesb)(x, y, robust, &dzero, &zero, iv, v);
        F77_SUB(lowese)(iv, v, n, x, surface);
        loess_prune(parameter, a, xi, vert, vval);
    }
    else if (!strcmp(*surf_stat, "direct/none")) {
        F77_SUB(lowesf)(x, y, robust, iv, v, n, x, &dzero, &zero, surface);
    }
    else if (!strcmp(*surf_stat, "interpolate/1.approx")) {
        F77_SUB(lowesb)(x, y, weights, diagonal, &one, iv, v);
        F77_SUB(lowese)(iv, v, n, x, surface);
        nsing = iv[29];
        for(i = 0; i < (*n); i++) *trL = *trL + diagonal[i];
        F77_SUB(lowesa)(trL, n, d, &tau, &nsing, one_delta, two_delta);
        loess_prune(parameter, a, xi, vert, vval);
    }
    else if (!strcmp(*surf_stat, "interpolate/2.approx")) {
        F77_SUB(lowesb)(x, y, weights, &dzero, &zero, iv, v);
        F77_SUB(lowese)(iv, v, n, x, surface);
        nsing = iv[29];
        F77_SUB(ehg196)(&tau, d, span, trL);
        F77_SUB(lowesa)(trL, n, d, &tau, &nsing, one_delta, two_delta);
        loess_prune(parameter, a, xi, vert, vval);
    }
    else if (!strcmp(*surf_stat, "direct/approximate")) {
        F77_SUB(lowesf)(x, y, weights, iv, v, n, x, diagonal, &one, surface);
        nsing = iv[29];
        for(i = 0; i < (*n); i++) *trL = *trL + diagonal[i];
        F77_SUB(lowesa)(trL, n, d, &tau, &nsing, one_delta, two_delta);
    }
    else if (!strcmp(*surf_stat, "interpolate/exact")) {
        hat_matrix = Calloc((*n)*(*n), double);
        LL = Calloc((*n)*(*n), double);
        F77_SUB(lowesb)(x, y, weights, diagonal, &one, iv, v);
        F77_SUB(lowesl)(iv, v, n, x, hat_matrix);
        F77_SUB(lowesc)(n, hat_matrix, LL, trL, one_delta, two_delta);
        F77_SUB(lowese)(iv, v, n, x, surface);
        loess_prune(parameter, a, xi, vert, vval);
        Free(hat_matrix);
        Free(LL);

    }
    else if (!strcmp(*surf_stat, "direct/exact")) {
        hat_matrix = Calloc((*n)*(*n), double);
        LL = Calloc((*n)*(*n), double);
        F77_SUB(lowesf)(x, y, weights, iv, v, n, x, hat_matrix, &two, surface);
        F77_SUB(lowesc)(n, hat_matrix, LL, trL, one_delta, two_delta);
        k = (*n) + 1;
        for(i = 0; i < (*n); i++)
            diagonal[i] = hat_matrix[i * k];
        Free(hat_matrix);
        Free(LL);
    }
    loess_free();
}

void
loess_dfit(double *y, double *x, double *x_evaluate, double *weights,
           double *span, int *degree, int *nonparametric,
           int *drop_square, int *sum_drop_sqr, int *d, int *n, int *m,
           double *fit)
{
    int zero = 0;
    double dzero = 0.0;
    loess_workspace(d, n, span, degree, nonparametric, drop_square,
                    sum_drop_sqr, &zero);
    F77_SUB(lowesf)(x, y, weights, iv, v, m, x_evaluate, &dzero, &zero, fit);
    loess_free();
}

void
loess_dfitse(double *y, double *x, double *x_evaluate, double *weights,
             double *robust, int *family, double *span, int *degree,
             int *nonparametric, int *drop_square, int *sum_drop_sqr,
             int *d, int *n, int *m, double *fit, double *L)
{
    int zero = 0, two = 2;
    double dzero = 0.0;
    loess_workspace(d, n, span, degree, nonparametric, drop_square,
                    sum_drop_sqr, &zero);
    if(*family == GAUSSIAN)
        F77_SUB(lowesf)(x, y, weights, iv, v, m, x_evaluate, L, &two, fit);
    else if(*family == SYMMETRIC)
    {
        F77_SUB(lowesf)(x, y, weights, iv, v, m, x_evaluate, L, &two, fit);
        F77_SUB(lowesf)(x, y, robust, iv, v, m, x_evaluate, &dzero, &zero, fit);
    }
    loess_free();
}

void
loess_ifit(int *parameter, int *a, double *xi, double *vert, double *vval,
           int *m, double *x_evaluate, double *fit)
{
    loess_grow(parameter, a, xi, vert, vval);
    F77_SUB(lowese)(iv, v, m, x_evaluate, fit);
    loess_free();
}

void
loess_ise(double *y, double *x, double *x_evaluate, double *weights,
          double *span, int *degree, int *nonparametric, int *drop_square,
          int *sum_drop_sqr, double *cell, int *d, int *n, int *m,
          double *fit, double *L)
{
    int zero = 0, one = 1;
    double dzero = 0.0;
    loess_workspace(d, n, span, degree, nonparametric, drop_square,
                   sum_drop_sqr, &one);
    v[1] = *cell;
    F77_SUB(lowesb)(x, y, weights, &dzero, &zero, iv, v);
    F77_SUB(lowesl)(iv, v, m, x_evaluate, L);
    loess_free();
}

// Set global variables  tau, lv, liv , and allocate global arrays
// v[1..lv],  iv[1..liv]
static void
loess_workspace(int *d, int *n, double *span, int *degree,
                int *nonparametric, int *drop_square, int *sum_drop_sqr,
                int *setLf)
{
    int D, N, tau0, nvmax, nf, i;
    double dlv, dliv;

    D = *d;
    N = *n;
    nvmax = MAX(200, N);
    nf = MIN(N, (int) floor(N * (*span) + 1e-5));
    if (nf <= 0) {
    	error_status = 1;
    	error_message = "span is too small";
        return;
    }
    tau0 = ((*degree) > 1) ? ((D + 2) * (D + 1) / 2) : (D + 1);
    tau = tau0 - (*sum_drop_sqr);
    dlv = 50 + (3 * D + 3) * nvmax + N + (tau0 + 2) * nf;
    dliv = 50 + (pow(2.0, (double) D) + 4.0) * nvmax + 2.0 * N;

    if (*setLf) {
        dlv = dlv + (D + 1.0) * nf * (double) nvmax;
        dliv = dliv + nf * (double) nvmax;
    }

    if (dlv < INT_MAX && dliv < INT_MAX) {
        lv  = (int) dlv;
        liv = (int) dliv;
    } else {
    	error_status = 1;
    	error_message = "workspace required is too large";
        return;
    }

    iv = Calloc(liv, int);
    v = Calloc(lv, double);

    F77_SUB(lowesd)(iv, &liv, &lv, v, d, n, span, degree, &nf,
                    &nvmax, setLf);
    iv[32] = *nonparametric;
    for(i = 0; i < D; i++)
        iv[i + 40] = drop_square[i];
}

static void
loess_prune(int *parameter, int *a, double *xi, double *vert, double *vval)
{
    int d, vc, a1, v1, xi1, vv1, nc, nv, nvmax, i, k;

    d = iv[1];
    vc = iv[3] - 1;
    nc = iv[4];
    nv = iv[5];
    a1 = iv[6] - 1;
    v1 = iv[10] - 1;
    xi1 = iv[11] - 1;
    vv1 = iv[12] - 1;
    nvmax = iv[13];

    for(i = 0; i < 5; i++)
        parameter[i] = iv[i + 1];
    parameter[5] = iv[21] - 1;
    parameter[6] = iv[14] - 1;

    for(i = 0; i < d; i++) {
        k = nvmax * i;
        vert[i] = v[v1 + k];
        vert[i + d] = v[v1 + vc + k];
    }
    for(i = 0; i < nc; i++) {
        xi[i] = v[xi1 + i];
        a[i] = iv[a1 + i];
    }
    k = (d + 1) * nv;
    for(i = 0; i < k; i++)
        vval[i] = v[vv1 + i];
}

static void
loess_grow(int *parameter, int *a, double *xi, double *vert, double *vval)
{
    int d, vc, nc, nv, a1, v1, xi1, vv1, i, k;

    d = parameter[0];
    vc = parameter[2];
    nc = parameter[3];
    nv = parameter[4];
    liv = parameter[5];
    lv = parameter[6];
    iv = Calloc(liv, int);
    v = Calloc(lv, double);

    iv[1] = d;
    iv[2] = parameter[1];
    iv[3] = vc;
    iv[5] = iv[13] = nv;
    iv[4] = iv[16] = nc;
    iv[6] = 50;
    iv[7] = iv[6] + nc;
    iv[8] = iv[7] + vc * nc;
    iv[9] = iv[8] + nc;
    iv[10] = 50;
    iv[12] = iv[10] + nv * d;
    iv[11] = iv[12] + (d + 1) * nv;
    iv[27] = 173;

    v1 = iv[10] - 1;
    xi1 = iv[11] - 1;
    a1 = iv[6] - 1;
    vv1 = iv[12] - 1;

    for(i = 0; i < d; i++) {
        k = nv * i;
        v[v1 + k] = vert[i];
        v[v1 + vc - 1 + k] = vert[i + d];
    }
    for(i = 0; i < nc; i++) {
        v[xi1 + i] = xi[i];
        iv[a1 + i] = a[i];
    }
    k = (d + 1) * nv;
    for(i = 0; i < k; i++)
        v[vv1 + i] = vval[i];

    F77_SUB(ehg169)(&d, &vc, &nc, &nc, &nv, &nv, v+v1, iv+a1,
                    v+xi1, iv+iv[7]-1, iv+iv[8]-1, iv+iv[9]-1);
}

static void
loess_free(void)
{
    Free(v);
    Free(iv);
}

/* begin ehg's FORTRAN-callable C-codes */

void
F77_SUB(ehg182)(int *i)
{
    char *mess, mess2[50];
    switch(*i){
        case 100:
            mess="Wrong version number in lowesd.  Probably typo in caller.";
            break;
        case 101:
            mess="d>dMAX in ehg131.  Need to recompile with increased dimensions.";
            break;
        case 102:
            mess="liv too small. (Discovered by lowesd)";
            break;
        case 103:
            mess="lv too small. (Discovered by lowesd)";
            break;
        case 104:
            mess="Span too small. Fewer data values than degrees of freedom.";
            break;
        case 105:
            mess="k>d2MAX in ehg136.  Need to recompile with increased dimensions.";
            break;
        case 106:
            mess="lwork too small";
            break;
        case 107:
            mess="Invalid value for kernel";
            break;
        case 108:
            mess="Invalid value for ideg";
            break;
        case 109:
            mess="lowstt only applies when kernel=1.";
            break;
        case 110:
            mess="Not enough extra workspace for robustness calculation";
            break;
        case 120:
            mess="Zero-width neighborhood. Make span bigger";
            break;
        case 121:
            mess="All data on boundary of neighborhood. make span bigger";
            break;
        case 122:
            mess="Extrapolation not allowed with blending";
            break;
        case 123:
            mess="ihat=1 (diag L) in l2fit only makes sense if z=x (eval=data).";
            break;
        case 171:
            mess="lowesd must be called first.";
            break;
        case 172:
            mess="lowesf must not come between lowesb and lowese, lowesr, or lowesl.";
            break;
        case 173:
            mess="lowesb must come before lowese, lowesr, or lowesl.";
            break;
        case 174:
            mess="lowesb need not be called twice.";
            break;
        case 175:
            mess="Need setLf=.true. for lowesl.";
            break;
        case 180:
            mess="nv>nvmax in cpvert.";
            break;
        case 181:
            mess="nt>20 in eval.";
            break;
        case 182:
            mess="svddc failed in l2fit.";
            break;
        case 183:
            mess="Did not find edge in vleaf.";
            break;
        case 184:
            mess="Zero-width cell found in vleaf.";
            break;
        case 185:
            mess="Trouble descending to leaf in vleaf.";
            break;
        case 186:
            mess="Insufficient workspace for lowesf.";
            break;
        case 187:
            mess="Insufficient stack space.";
            break;
        case 188:
            mess="lv too small for computing explicit L.";
            break;
        case 191:
            mess="Computed trace L was negative; something is wrong!";
            break;
        case 192:
            mess="Computed delta was negative; something is wrong!";
            break;
        case 193:
            mess="Workspace in loread appears to be corrupted.";
            break;
        case 194:
            mess="Trouble in l2fit/l2tr";
            break;
        case 195:
            mess="Only constant, linear, or quadratic local models allowed";
            break;
        case 196:
            mess="degree must be at least 1 for vertex influence matrix";
            break;
        case 999:
            mess="not yet implemented";
            break;
        default:
            sprintf(mess=mess2,"Assert failed; error code %d\n",*i);
            break;
    }
    error_status = 1;
    error_message = mess;
}

void F77_SUB(ehg183a)(char *s, int *nc, int *i, int *n,int *inc)
{
    char mess[4000], num[20];
    int j;
    strncpy(mess,s,*nc);
    mess[*nc] = '\0';
    for (j=0; j<*n; j++) {
	    snprintf(num, 20, " %d",i[j * *inc]);
	    strcat(mess,num);
    }
    strcat(mess,"\n");
    error_status = 1;
    error_message = mess;
}

void F77_SUB(ehg184a)(char *s, int *nc, double *x, int *n, int *inc)
{
    char mess[4000], num[30];
    int j;
    strncpy(mess,s,*nc);
    mess[*nc] = '\0';
    for (j=0; j<*n; j++) {
	    snprintf(num,30," %.5g",x[j * *inc]);
	    strcat(mess,num);
    }
    strcat(mess,"\n");
    error_status = 1;
    error_message = mess;
}
