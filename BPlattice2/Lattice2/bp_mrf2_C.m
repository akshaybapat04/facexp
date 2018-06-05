function [new_bel, niter] = bp_mrf2_C(pot, local_evidence, varargin)
% BP_MRF2 Belief propagation on a 2D lattice pairwise MRF in C
% function [bel, niter] = bp_mrf2_C(pot, local_evidence, varargin)
%
% Input:
% pot(k1,k2) = potential on edge between nodes i,j (must be the same for all pairs)
% local_evidence(r,c,j) = Pr(observation at node (r,c) | Xi=j); 3D matrix
%
% Output:
% bel(r,c,k) = P(X(r,c)=k|evidence)
% niter contains the number of iterations used 
%
% [ ... ] = bp_mrf2(..., 'param1',val1, 'param2',val2, ...)
% allows you to specify optional parameters as name/value pairs.
% Parameters names are below [default value in brackets]
%
% max_iter - max. num. iterations [ 5*nnodes]
% momentum - weight assigned to old message in convex combination
%            (useful for damping oscillations) - currently ignored i[0]
% tol      - tolerance used to assess convergence [1e-3]
% maximize - 1 means use max-product, 0 means use sum-product [0]
% verbose - 1 means print error at every iteration [0]

s = sum(pot,2);
if max(s-s(1))>1e-3
  error('kernel must have constant row sum')
end

[max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'max_iter', 500, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

[new_bel, niter]=bp_mrf2_Chelper(pot, local_evidence, maximize, max_iter, tol);

fprintf('converged in %d iterations\n', niter);
