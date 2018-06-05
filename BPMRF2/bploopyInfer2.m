function [bel, belE, logZ, msg, niter] = bploopyInfer(E, pot, localEv, nstates,varargin)
% BP_MRF2_GENERAL Belief propagation on an MRF with pairwise potentials
%
% Input:
% E(i,j) = edge number or 0 if absent
% pot{e}(ki,kj) = potential on edge between nodes i,j
% localEv{i}(k) = Pr(observation at node i | Xi=k)
%
%
% Output:
% bel{i}(k) = P(Xi=k|evidence)
% belE{e}(:) = P(edge e)
% neglogZ = Bethe approximation to -log(Z) 
% niter contains the number of iterations used 
%
% [ ... ] = bp_mrf2(..., 'param1',val1, 'param2',val2, ...)
% allows you to specify optional parameters as name/value pairs.
% Parameters names are below [default value in brackets]
%
% max_iter - max. num. iterations [ 5*nnodes]
% momentum - weight assigned to old message in convex combination
%            (useful for damping oscillations) - currently ignored i[0]
% tol      - tolerance used to assess convergence [1e-3]
% maximize - 1 means use max-product, 0 means use sum-product [0]
% verbose - 1 means print error at every iteration [0]
%
% fn - name of function to call at end of every iteration [ [] ]
% fnargs - we call feval(fn, bel, iter, fnargs{:}) [ [] ]
%tic;
%fprintf('Loopy,');


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




E = engine.E;
nstates = engine.nstates;
verbose = engine.verbose;
maximize = engine.maximize;

nnodes = length(E);
nedges = length(find(E))/2;
E = E;
adj_mat = (E>0);

% initialise messages
prod_of_msgs = cell(1, nnodes);
old_bel = cell(1, nnodes);
old_msg = cell(1, nedges);
for i=1:nnodes
  prod_of_msgs{i} = localEv{i};
  old_bel{i} = localEv{i};
  nbrs = find(adj_mat(:,i));
  for j=nbrs(:)'
    %fprintf('i=%d,j=%d,e=%d,s=%d\n', i, j, E(i,j), nstates(j));
    old_msg{E(i,j)} = normalise(ones(nstates(j),1));
  end
end


converged = 0;
iter = 1;
done = 0;

while ~converged & (iter <= engine.max_iter)
  
  % each node sends a msg to each of its neighbors
  for i=1:nnodes
    nbrs = find(adj_mat(i,:));
    for j=nbrs(:)'

      if i<j
	pot_ij = pot{E(i,j)};
      else
	pot_ij = pot{E(j,i)}';
      end
      pot_ij = pot_ij';
      % now pot_ij(xj, xi)  so pot_ij * msg(xi) = sum_xi pot(xj,xi) msg(xi) = f(xj)

      if 1
	% Compute temp = product of all incoming msgs except from j
	% by dividing out old msg from j from the product of all msgs sent to i
	temp = prod_of_msgs{i};
	m = old_msg{E(j,i)};
	if any(m==0)
	  fprintf('m=0, iter=%d, send from i=%d to j=%d\n', iter, i, j);
	  done=1;
	  break;
	  %keyboard
	end
	m = m + (m==0); % valid since m(k)=0 => temp(k)=0, so can replace 0's with anything
	temp = temp ./ m;
	temp_div = temp;
      end
      
      if ~done
	% Compute temp = product of all incoming msgs except from j in obvious way
	%temp = ones(nstates(i),1);
	temp = localEv{i};
	for k=nbrs(:)'
	  if k==j, continue, end;
	  temp = temp .* old_msg{E(k,i)};
	end
	%assert(approxeq(temp, temp_div))

	assert(approxeq(normalise(pot_ij * temp), normalise(pot_ij * temp_div)))
      end
      
      if maximize
	newm = max_mult(pot_ij, temp); % bottleneck
      else
	newm = pot_ij * temp;
      end
      newm = normalise(newm);
      new_msg{E(i,j)} = newm;
    end % for j 
  end % for i
  old_prod_of_msgs = prod_of_msgs;
  
  % each node multiplies all its incoming msgs and computes its local belief
  for i=1:nnodes
    nbrs = find(adj_mat(:,i));
    prod_of_msgs{i} = localEv{i};
    for j=nbrs(:)'
      prod_of_msgs{i} = prod_of_msgs{i} .* new_msg{E(j,i)};
    end
    new_bel{i} = normalise(prod_of_msgs{i});
  end
  err = abs(cat(1,new_bel{:}) - cat(1, old_bel{:}));

  converged = all(err < engine.tol) | done;
  if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
  %celldisp(new_bel)
  if ~isempty(engine.fn)
    if isempty(engine.fnargs)
      feval(engine.fn, new_bel);
    else
      feval(engine.fn, new_bel, iter, engine.fnargs{:});
    end
  end
  
  iter = iter + 1;
  old_msg = new_msg;
  old_bel = new_bel;
end % while

niter = iter-1;

if verbose, fprintf('converged in %d iterations\n', niter); end

bel = new_bel;
msg = new_msg;

if nargout >= 2,
  belE = computeEdgeBel(E, pot, bel, msg);
end

if nargout >= 3,
  neglogZ = betheMRF2(E, pot, localEv, bel, belE);
  logZ = -neglogZ;
end
%fprintf('%d its\n',niter);
