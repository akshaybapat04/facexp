#include <stdio.h>
#include "mex.h"

/* Same as kronSum.m but faster */

/* A(i,j) = A(i + nrows*j) since Matlab uses Fortran layout. */

 
#define b2(q,t) b[(q)+Q*(t)]
#define f2(d,t) f[(d)+D*(t)]


void mexFunction(
                 int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]
		 )
{
  double *Fout, *b, *f;
  int D, Q, T, d, q, t, i;
  
  b = mxGetPr(prhs[0]);
  Q = mxGetM(prhs[0]);
  T = mxGetN(prhs[0]);

  f = mxGetPr(prhs[1]);
  D = mxGetM(prhs[1]);

  plhs[0] = mxCreateDoubleMatrix(D*Q, 1, mxREAL);
  Fout = mxGetPr(plhs[0]); 

  for (t=0; t < T; t++) {
    i = 0;
    for (q = 0; q < Q; q++) {
      for (d = 0; d < D; d++) {
	Fout[i] += b2(q,t) * f2(d,t);
	i++;
      }
    }
  }

}



