function net = crfchain(inputDims, nstates,  varargin)
% CRFCHAIN make a 1D conditional random field structure
% function net = crf(inputDims, nstates, ...)
%
% inputDims is the size of the observed feature vector 
% nstates is the number of discrete hiddden states 
%
% Optional arguments passed as name/value pairs
%
% 'alpha' - weight to put on ||w^2|| regularizer term (alpha=1/sigma^2)
%
% 'clampedWeightsForOneState' - if 1, the number of free parameters
%   for each private evidence is reduced, to avoid the sum-to-one degeneracy
%   (eg in the 2-class case, only one weight vector is needed to specify the
%   decision boundary).



[net.alpha, net.clampWeightsForOneState] = ...
    process_options(...
    varargin, 'alpha', 0.1, 'clampWeightsForOneState', 0);

assert(net.clampWeightsForOneState==0)

net.type = 'crfchain';
net.nin = []; 
net.nout = [];

net.inputDims = inputDims;
net.nstates = nstates;

net.w = 0.1*randn(inputDims, nstates);
net.pot = normalize(rand(nstates, nstates));

net.nparams = inputDims*nstates + nstates*nstates;

net.infEngine.maximize = 0;
