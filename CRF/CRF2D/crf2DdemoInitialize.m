function [netOrig] = mk_2D_latticeCRF(nr,nc,nstates)
% Sets the parameters and calls the constructor of a 2D lattice CRF
% (with tied parameters and one state kept constant)

Nnodes = nr*nc;
Nedges = (nr-1)*nc + nr*(nc-1);
G = mk_2D_lattice(nr, nc, 4); 
D = 1; % scalar observations

seed = 1;
rand('state', seed); randn('state', seed);
eclassEdge = ones(1,Nedges);
adjustableEdge = 1;
netOrig = crf(repmat(D,1,Nnodes), repmat(nstates,1,Nnodes), G, 'eclassNode', ones(1,Nnodes), ...
    'eclassEdge', eclassEdge, 'alpha', 0, 'clampWeightsForOneState', 1, ...
    'addOneToFeatures', 1, 'adjustableEdgeEclassBitv', [adjustableEdge]);