function [mpe, niter, bel] = bp_mpe_mrf2_general(adj_mat, pot, local_evidence, varargin)
% BP_MPE_MRF2_GENERAL MAP estimate for pairwise MRF with arbitrary structure
% function [mpe, niter, bel] = bp_mpe_mrf2_general(adj_mat, pot, local_evidence, varargin)
%
% Inputs: same as bp_mrf2, except maximize is always 1.
% local_evidence can either be a cell array, local_evidence{i}(k), or, if all
% hidden nodes have the same number of states, a regular array, local_evidence(k,i),
% where k indexes states and i indexes nodes.
%
% Output:
% mpe(i) is the marginally most probable value of i (thresholding the max-product belief)
%   This may give inconsistent results if there are ties.

% We should use max-product; sum-product is slightly faster, but often takes more
% iterations to converge.
[bel, niter] = bp_mrf2_general(adj_mat, pot, local_evidence, 'maximize', 1, varargin{:});

mpe = bel_to_mpe(bel);




