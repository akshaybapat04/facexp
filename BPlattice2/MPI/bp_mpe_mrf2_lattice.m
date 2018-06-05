function [mpe, niter] = bp_mpe_mrf2_lattice(pot, local_evidence, varargin)
% BP_MPE_MRF2 Find the most probable explanation of the data for an MRF2 using loopy bel prop
% function [mpe, niter] = bp_mpe_mrf2_lattice(pot, local_evidence, varargin)
%
% INPUT:
% pot: KxK array, where K = nstates
% local_evidence(r,c,k) is an nrows x ncols x K array
% optional arguments - same as bp_mrf2_lattice, plus
%
% 'method' - 'vectorized', 'forloops1', 'strips', 'local' [vectorized]
%
% Output:
% mpe(r,c) is the marginally most probable value of pixel r,c (thresholding the max-product belief)

% We should run max-product and then threshold.
% This may give inconsistent results if there are ties.
% Sum-product is slightly faster, but often takes more
% iterations to converge.

[method, other_args] = process_options(varargin, 'method', 'vectorized');

fname = sprintf('bp_mrf2_lattice_%s', method);
tic;
[bel, niter] = feval(fname, pot, local_evidence, 'maximize', 1, other_args{:});
toc;
  
[nrows ncols nstates] = size(local_evidence);
nnodes = nrows*ncols;

bel=reshape(bel, [nnodes nstates]);
mpe = reshape(bel_to_mpe(bel'), [nrows ncols]);
