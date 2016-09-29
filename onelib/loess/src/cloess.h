#include <stdio.h>
#include <stdlib.h>


// from loess.c
void loess_model_setup(loess_model *model);
void loess_inputs_setup(double *x, double *y, double *w, long n, long p, loess_inputs *inputs);
void loess_outputs_setup(long n, long p, loess_outputs *outputs);
void loess_control_setup(loess_control *control);
void loess_kd_tree_setup(long n, long p, loess_kd_tree *kd_tree);
void loess_setup(double *x, double *y, double *w, int n, int p, loess *lo);
char loess_fit(loess *lo);
void loess_free_mem(loess *lo);
void loess_summary(loess *lo);
               
// from misc.c
void pointwise(prediction *pre, int m, double coverage, conf_inv *ci); 
void pw_free_mem(conf_inv *ci);
double pf(double q, double df1, double df2);
double ibeta(double x, double a, double b);
////
// from predict.c
void predict(double *eval, int m, loess *lo, prediction *pre);
void pred_free_mem(prediction *pre);
//

