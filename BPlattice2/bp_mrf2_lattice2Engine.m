function engine = lattice2hmmCellEngine(E,G,nr, nc, nstates, maxIter, varargin)
% LATTICE2HMMCELLENGINE Exact infernece on a 2D MRF lattice by converting to an 1D chain (HMM)
% function engine = lattice2hmmCellEngine(varargin)
%
% This is just a front-end to the lattice2_hmm code, which uses cell arrays for input/output
% so it can be used with the general CRF code.

engine.E = E;
engine.G = G;
engine.type = 'bp_mrf2_lattice2';
engine.nrows = nr;
engine.ncols = nc;
engine.nstates = nstates;
engine.maxIter = maxIter;

