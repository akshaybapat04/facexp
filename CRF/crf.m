function net = crf(inputDims, nstates, G, varargin)
% CRF make a conditional random field structure
% function net = crf(inputDims, nstates, G, ...)
%
% inputDims(i) is the size of the feature vector for node i
% nstates(i) is the number of discrete states for node i
% G(i,j) = 0/1 is an adjacency matrix for the graph on the nodes
% infoAlgo is a string containing the name of a function to do inference;
%    this must be a function of the same form as 'bptreeEngine/Infer'.
%
% A CRF is a way of classifying multiple objects/targets y jointly
% using the following distribution
%   P(y(1:dy) | x(:,1:dx)) = 1/Z exp[ E(y,x) ]
% where 
%   E(y,x) = sum_i w{i}(:,y(i))^T * x{i}(:) + sum_{e} log phi{e}(y(i), y(j)) 
%     is the 'energy' associated with configuration y given input x
%   x{i}(:) is the feature vector for node i of size inputDims(i)
%   w{i}(:,q) is the weight vector for node i and state q in [1..nstates(i)]
%   e = edge i-j in G
%   phi{e}(qi,qj) is a potential function (table of constraints)   where i<j
%   Z = sum_{y'} exp E(y',x) is the partition function (normalization term)
%
% x and w are cell arrays to allow each node to have a different sized feature vector.
% phi is a cell array to allow each node to have a different number of states.
%
% Optional arguments passed as name/value pairs
% 'infEngineName' - 'bptree'
%
% 'alpha' - weight to put on ||w^2|| regularizer term (alpha=1/sigma^2)
%
% 'clampedWeightsForOneState' - if 1, the number of free parameters
%   for each private evidence is reduced, to avoid the sum-to-one degeneracy
%   (eg in the 2-class case, only one weight vector is needed to specify the
%   decision boundary).
%
% 'addOneToFeatures' - if 1, we append 1 to each observed feature vector
%
% 'eclassNode' - if eclassNode(i)=eclassNode(j)=e, then the parameters
%   for the local evidences for nodes i,j are shared (tied).
%   By default, eclassNode = 1:N, meaning each node is its own equivalence class.
%
% 'eclassEdge' - analogout to eclassNode, but for tying edge potentials.


net.G = G;
[net.E, net.nedges] = assignEdgeNums(G);
net.nnodes = length(nstates);

[net.adjustableEdgeEclassBitv, net.addOneToFeatures, net.alpha, net.clampWeightsForOneState,...
 net.eclassNode, net.eclassEdge] = ...
    process_options(...
    varargin, 'adjustableEdgeEclassBitv', [], ...
	'addOneToFeatures', 0, 'alpha', 0.1, 'clampWeightsForOneState', 0, ...
	'eclassNode', 1:net.nnodes, 'eclassEdge', 1:net.nedges);

net.type = 'crf';
net.nin = length(inputDims);
net.nout = length(nstates);

if net.addOneToFeatures
  inputDims = inputDims+1;
end
net.inputDims = inputDims;
net.nstates = nstates;

net.nnodeEclasses = max(net.eclassNode);
net.w = cell(1, net.nnodeEclasses);
net.nedgeEclasses = max(net.eclassEdge);
net.pot = cell(1, net.nedgeEclasses);

if isempty(net.adjustableEdgeEclassBitv)
  net.adjustableEdgeEclassBitv = ones(1, net.nedgeEclasses);
end

for ec=1:net.nnodeEclasses
  ndx = find(net.eclassNode==ec);
  net.representativeNode(ec) = ndx(1);
end
for ec=1:net.nedgeEclasses
  ndx = find(net.eclassEdge==ec);
  [i,j]=find(net.E==ndx(1));
  net.representativeEdge(ec,:) = [i j];
end

net = crfInitParams(net);

net.nparams = sum(net.nparamsPerNodeEclass) + sum(net.nstatesPerEdgeEclass);
adjustableEdges = find(net.adjustableEdgeEclassBitv);
net.nparamsAdjustable = sum(net.nparamsPerNodeEclass) + ...
    sum(net.nstatesPerEdgeEclass(adjustableEdges));

% default, may be overwritten
%net.infEngineName = 'bptree';
%net.infEngine = bptreeEngine(net.E, net.nstates);
