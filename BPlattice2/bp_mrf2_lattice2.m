function [bel, niter, msgs] = bp_mrf2_lattice2(pot, local_evidence, varargin)
% BP_MRF2_LATTICE2 Belief propagation on a 2D lattice with pairwise potentials
% function [bel, niter, msgs] = bp_mrf2_lattice2(pot, local_evidence, varargin)
%
% INPUT:
% pot(k1,k2) = potential on edge between nodes i,j (must be the same for all pairs)
% local_evidence(r,c,j) = Pr(observation at node (r,c) | hidden state j)
%
% OUTPUT:
% bel(r,c,j) = P(X(r,c)=j|evidence)
% niter contains the number of iterations used 
%
% [ ... ] = bp_mrf2(..., 'param1',val1, 'param2',val2, ...)
% allows you to specify parameters as name/value pairs.
% Parameters names are below [default value in brackets]
%
% Required:
% method - 'vectorized', 'forloops', 'strips', 'local', 'C' [vectorized]
%
% If method = strips or local, you must specify
% nstrips - number of parallel columns to divide image into for simultaneous processing [1]
%
% Optional parameters shared by all methods:
% max_iter - max. num. iterations [ 5*nnodes]
% momentum - weight assigned to old message in convex combination
%            (useful for damping oscillations) - currently ignored [0]
% tol      - tolerance used to assess convergence [1e-3]
% maximize - 1 means use max-product, 0 means use sum-product [0]
% verbose - 1 means print error at every iteration [0]
%
% Examples:
% [bel, niter] = bp_mrf2_lattice2(pot, local_evidence);
% [bel, niter] = bp_mrf2_lattice2(pot, local_evidence, 'method', 'strips', 'nstrips', 2);


[method, other_args] = process_options(varargin, 'method', 'vectorized');

msgs = [];
%fname = sprintf('bp_mrf2_lattice_%s', method);
fname = sprintf('bp_mrf2_%s', method);
[bel, niter, msgs] = feval(fname, pot, local_evidence, other_args{:});
