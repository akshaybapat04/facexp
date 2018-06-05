function [bel, niter] = bp_mrf2_lattice_mpi(pot, local_evidence, varargin)
% Setup function for parallel version.
%
% BP_MRF2 Belief propagation on an MRF with pairwise potentials
% function [bel, niter] = bp_mrf2_lattice(pot, local_evidence, varargin)
%
% This is a modified version of bp_mrf2_lattice_strips,
% designed to be easy to parallelize.
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
% nstrips - number of parallel columns to divide image into for simultaneous processing [1]
% max_iter - max. num. iterations [ 5*nnodes]
% momentum - weight assigned to old message in convex combination
%            (useful for damping oscillations) - currently ignored [0]
% tol      - tolerance used to assess convergence [1e-3]
% maximize - 1 means use max-product, 0 means use sum-product [0]
% verbose - 1 means print error at every iteration [0]


[nrows ncols nstates] = size(local_evidence);

nnodes=nrows*ncols;
ndir = 4;

[nstrips, max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'nstrips', 1, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

ncols_per_strip = ncols/nstrips;
if mod(ncols, nstrips)>0
  error(sprintf('nstrips %d must be a multiple of ncols %d', nstrips, ncols))
end

% Get a list of machines.
cpus = get_machines;

% Launch MatlabMPI job.
eval( MPI_Run('bp_mrf2_lattice_local_mpi',nstrips,cpus) );
