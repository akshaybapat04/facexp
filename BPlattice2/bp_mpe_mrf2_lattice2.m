function [mpe, niter] = bp_mpe_mrf2_lattice2(pot, local_evidence, varargin)
% BP_MPE_MRF2 Find the most probable explanation of the data for a 2D lattice MRF2
% function [mpe, niter] = bp_mpe_mrf2_lattice2(pot, local_evidence, varargin)
%
% INPUT:
% pot: KxK array, where K = nstates
% local_evidence(r,c,k) is an nrows x ncols x K array
% optional arguments - same as bp_mrf2_lattice2
%
% Output:
% mpe(r,c) is the marginally most probable value of pixel r,c (thresholding the max-product belief)

% We should run max-product and then threshold.
% This may give inconsistent results if there are ties.
% Sum-product is slightly faster, but often takes more
% iterations to converge.

[method, other_args] = process_options(varargin, 'method', 'vectorized');

%fname = sprintf('bp_mrf2_lattice_%s', method);
fname = sprintf('bp_mrf2_%s', method);
[bel, niter] = feval(fname, pot, local_evidence, 'maximize', 1, other_args{:});
  
[nrows ncols nstates] = size(local_evidence);
nnodes = nrows*ncols;

bel=reshape(bel, [nnodes nstates]);
mpe = reshape(bel_to_mpe(bel'), [nrows ncols]);
