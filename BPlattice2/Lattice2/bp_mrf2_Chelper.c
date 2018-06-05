#include <math.h>
#include "mex.h"

double max(double a, double b)
{
  if (a>b)
    return a;
  else return b;
}

/* call with potential, local_evidence, max_mult, max_iter, tol */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  
  int rows;
  int cols;
  int nstates;
  int max_mult;
  int nnodes;
  int dims[3];
  int dims2[2]={1,1};
  int i, j, k, l, m; /* indices */

  int converged=0;
  int niter=0;
  int max_iter;

  double tol;

  double prod, sum, error, temp; /* temporary variables */

  
  double *vector; /* temporary vector of size nstates */
  
  double *iter;
  double *new_bel;
  double **prod_of_msgs, **msgs, **old_msgs;
  /* 3 and 4 dimensional arrays (note: new_bel treated differently than
     others since it must be returned
     other way saves more memory */

  double *loc, *pot;
  /* hold input arrays */

  if (nrhs<2 || nlhs!=2)
    mexErrMsgTxt("bp_mrf2_lattice_vectorized requires two inputs and two outputs");
  if (mxIsChar(prhs[0])||mxIsClass(prhs[0], "sparse")||mxIsComplex(prhs[0])||mxIsChar(prhs[1])||mxIsClass(prhs[1], "sparse")||mxIsComplex(prhs[1]))
    mexErrMsgTxt("Inputs must be real, full, and nonstring");
  if (mxGetN(prhs[0])!=mxGetM(prhs[0]))
    mexErrMsgTxt("Potential must be square");
  if (mxGetNumberOfDimensions(prhs[1])!=3)
    mexErrMsgTxt("local evidence must be a three dimensional array");
  if (mxGetDimensions(prhs[1])[2]!=mxGetN(prhs[0]))
    mexErrMsgTxt("Potential and local evidence must have compatible dimensions");

  /* set up input variables */
  rows=mxGetDimensions(prhs[1])[0];
  cols=mxGetDimensions(prhs[1])[1];
  nstates=mxGetDimensions(prhs[1])[2];
  max_mult=mxGetPr(prhs[2])[0];
  max_iter=mxGetPr(prhs[3])[0];
  tol=mxGetPr(prhs[4])[0];
  dims[0]=rows;
  dims[1]=cols;
  dims[2]=nstates;

  /* set up new_bel and niter */
  plhs[0]=mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  new_bel=mxGetPr(plhs[0]);

  plhs[1]=mxCreateNumericArray(2, dims2, mxDOUBLE_CLASS, mxREAL);
  iter=mxGetPr(plhs[1]);

  vector=mxCalloc(nstates, sizeof(double));

  prod_of_msgs=mxCalloc(rows, sizeof(double*));
  msgs=mxCalloc(rows, sizeof(double*));
  old_msgs=mxCalloc(rows, sizeof(double*));

  pot=mxGetPr(prhs[0]);
  loc=mxGetPr(prhs[1]);

  for(i=0;i<rows;i++)
  {
    prod_of_msgs[i]=mxCalloc(cols*nstates, sizeof(double));
    msgs[i]=mxCalloc(cols*nstates*4, sizeof(double));
    old_msgs[i]=mxCalloc(cols*nstates*4, sizeof(double));
    for(j=0;j<cols;j++)
    {
      for(k=0;k<nstates;k++)
      {
	/* now, initialize prod and new_bel
	   again note differences in indexing */
	new_bel[i+rows*(j+cols*k)]=loc[i+rows*(j+cols*k)];
	prod_of_msgs[i][j+cols*k]=loc[i+rows*(j+cols*k)];
	
	/* now, initialize msgs, old_msgs */
	for(l=0;l<4;l++)
	{
	  msgs[i][j+cols*(k+nstates*l)]=1./(double)nstates;
	}
      }
    }
  }
  

  while ((!converged)&&(niter<=max_iter))
  {
    /* set old_msgs */
    for(i=0;i<rows;i++)
      for(j=0;j<cols;j++)
	for(k=0;k<nstates;k++)
	{
	  for(l=0;l<4;l++)
	    old_msgs[i][j+cols*(k+nstates*l)]=msgs[i][j+cols*(k+nstates*l)]+(msgs[i][j+cols*(k+nstates*l)]==0);
	}

    /* compute msgs */
    for(i=0;i<rows;i++)
    {
      for(j=0;j<cols;j++)
      {
	for(k=0;k<nstates;k++)
	{
	  /* new message is just old product divided by old message in */
	  /* opposite direction */

	  /* north message */
	  if (j>0) /* we have a northern neighbor */
	    msgs[i][j+cols*k]=prod_of_msgs[i][j-1+cols*k]/old_msgs[i][j-1+cols*(k+nstates*2)];
	  else
	    msgs[i][j+cols*k]=1;

	  /* south message */
	  if (j<cols-1)
	    msgs[i][j+cols*(k+nstates*2)]=prod_of_msgs[i][j+1+cols*k]/old_msgs[i][j+1+cols*k];
	  else
	    msgs[i][j+cols*(k+nstates*2)]=1;
	  
	  /* east message */
	  if (i>0)
	    msgs[i][j+cols*(k+nstates)]=prod_of_msgs[i-1][j+cols*k]/old_msgs[i-1][j+cols*(k+nstates*3)];
	  else
	    msgs[i][j+cols*(k+nstates)]=1;
	  
	  /* west message */
	  if (i<rows-1)
	    msgs[i][j+cols*(k+nstates*3)]=prod_of_msgs[i+1][j+cols*k]/old_msgs[i+1][j+cols*(k+nstates)];
	  else
	    msgs[i][j+cols*(k+nstates*3)]=1;
	}

	
	/* now multiply by pot */
	for (k=0; k<4; k++)
	{
	  sum=0; /* normalizing constant */

	  /* multiply one vector at a time */
	  /* vector stores the result of pot*msgs(i,j,:,k) */
	  for (l=0; l<nstates; l++)
	  {
	    vector[l]=0;
	    /* cycle through each col of pot and each row of msgs(i,j,:,k) */
	    /* if max_mult=1, calculate max product, else sum product */
	    for (m=0; m<nstates; m++)
	    {
	      if (max_mult)
		vector[l]=max(vector[l], pot[l+nstates*m]*msgs[i][j+cols*(m+nstates*k)]);
	      else
		vector[l]+=pot[l+nstates*m]*msgs[i][j+cols*(m+nstates*k)];
	    }
	    sum+=vector[l];
	  }
	  

	  /* now we can update msgs
	     note: we couldn't do this in the other loop, since msgs's
	     contents were required for further calculations */
	  for (l=0; l<nstates; l++)
	    msgs[i][j+cols*(l+nstates*k)]=vector[l]/sum;
	}
      }
    }

    error=0;
 
    /* now update prod_of_msgs and new_bel*/
    for(i=0; i<rows; i++)
    {
      for(j=0; j<cols; j++)
      {
	sum=0; /* for normalization */
	for(k=0; k<nstates; k++)
	{
	  prod=1;
	  for(l=0; l<4; l++)
	  {
	    prod*=msgs[i][j+cols*(k+nstates*l)];
	  }
	  prod_of_msgs[i][j+cols*k]=prod*loc[i+rows*(j+cols*k)];
	  sum+=prod_of_msgs[i][j+cols*k];
	}

	/* new_bel is just normalized prod_of_msgs */
	for(k=0; k<nstates; k++)
	{
	  /* calculate the error */
	  temp=prod_of_msgs[i][j+cols*k]/sum-new_bel[i+rows*(j+cols*k)];
	  error=max(temp,max(-temp,error));

	  /* set new_bel*/
	  new_bel[i+rows*(j+cols*k)]=prod_of_msgs[i][j+cols*k]/sum;
	}
      }
    }

    converged=(error<tol);
    niter++;
    iter[0]=(double)niter;
  }

  /* now free memory that we used */
  for(i=0; i<rows; i++)
  {
    mxFree(msgs[i]);
    mxFree(old_msgs[i]);
    mxFree(prod_of_msgs[i]);
  }
  mxFree(msgs);
  mxFree(old_msgs);
  mxFree(prod_of_msgs);
  mxFree(vector);
}
