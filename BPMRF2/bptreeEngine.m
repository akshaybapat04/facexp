function engine = bptreeEngine(E, nstates, varargin)
% bpmrf2TreeEngine Make structure for belief propagation on a tree
% function engine = bpmrf2TreeEngine(E, nstates, varargin)

engine.type = 'bptree';
nnodes = length(E);
Nedges = length(find(E));
msg = cell(1,Nedges); % bi-directional
edge_id = E;
adj_mat = (E>0);

root = nnodes;
[maximize, verbose, root] = ...
    process_options(varargin,  'maximize', 0, 'verbose', 0, ...
		    'root', root);

engine.maximize = maximize;
engine.verbose = verbose;
engine.E = E;
engine.nstates = nstates;

% determine order to send msgs
[engine.tree, engine.preorder, engine.postorder, cycle] = mk_rooted_tree(adj_mat, root);
if cycle
  error('bpmrf2Tree cannot handle cyclic graphs')
end





