function engine = bploopyEngine(E, nstates, varargin)
% function engine = loopybpEngine(E, nstates, varargin)


engine.type = 'bploopy';
nnodes = length(E);
Nedges = length(find(E));
msg = cell(1,Nedges); % bi-directional
edge_id = E;
adj_mat = (E>0);


[max_iter, momentum, tol, maximize, verbose, fn, fnargs] = ...
    process_options(varargin, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0, ...
		    'fn', [], 'fnargs', []);

engine.maximize = maximize;
engine.verbose = verbose;
engine.E = E;
engine.max_iter = max_iter;
engine.nstates = nstates;
engine.tol = tol;
engine.fn = fn;
engine.fnargs = fnargs;
