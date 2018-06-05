function engine = bpchainEngine(varargin)
% bpchainEngine Make structure for belief propagation on a 1D chain
% function engine = bpchainEngine(varargin)

engine.type = 'bpchain';

[maximize, verbose] = ...
    process_options(varargin,  'maximize', 0, 'verbose', 0);

engine.maximize = maximize;
engine.verbose = verbose;

