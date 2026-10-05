/*
 * The few libf2c routines that the C translation of the Fortran sources
 * calls. They are only needed when building for Emscripten (Pyodide), where
 * the Fortran is translated with f2c and there is no libf2c to link against.
 * Each follows the behaviour of the netlib libf2c routine of the same name.
 */
#include <math.h>

double
pow_dd(double *ap, double *bp)
{
    return pow(*ap, *bp);
}

int
pow_ii(int *ap, int *bp)
{
    int x = *ap, n = *bp, result = 1;
    unsigned int u;

    if (n <= 0) {
        if (n == 0 || x == 1)
            return 1;
        if (x != -1)
            return x == 0 ? 1 / x : 0;
        n = -n;
    }
    for (u = n; ; ) {
        if (u & 01)
            result *= x;
        if (u >>= 1)
            x *= x;
        else
            break;
    }
    return result;
}

double
d_sign(double *a, double *b)
{
    double x = (*a >= 0 ? *a : -*a);
    return (*b >= 0 ? x : -x);
}

int
i_len(char *s, int n)
{
    return n;
}
