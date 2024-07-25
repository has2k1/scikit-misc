#include <stdio.h>
#include <math.h>

#define Calloc(n,t)	(t *)calloc((unsigned)(n),sizeof(t))
#define Free(p)		free((char *)(p))

/* the mapping from f77 to C intermediate code -- may be machine dependent
 * the first definition satisfies lint's narrowminded preprocessing & should
 * stay the same for all implementations.  The __STDC__ definition is for
 * ANSI standard conforming C compilers. The #else definition should
 * generate the version of the fortran subroutine & common block names x
 * handed to the local loader; e.g., "x_" in system V, Berkeley & 9th edition
 */

#ifdef _MSC_VER
  // Microsoft Visual C++
  #define F77_SUB(x) x##_
#else
  // Other compilers
  #ifdef __STDC__
    #define F77_SUB(x) x##_
  #else
    #define F77_SUB(x) x/**/_
  #endif
#endif

#define NULL_ENTRY          ((int *)NULL)
