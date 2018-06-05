function engine = lattice2hmmCellEngine(nr, nc, nstates, varargin)
% LATTICE2HMMCELLENGINE Exact infernece on a 2D MRF lattice by converting to an 1D chain (HMM)
% function engine = lattice2hmmCellEngine(varargin)
%
% This is just a front-end to the lattice2_hmm code, which uses cell arrays for input/output
% so it can be used with the general CRF code.

engine.type = 'lattice2hmmCell';
engine.nrows = nr;
engine.ncols = nc;
engine.nstates = nstates;
