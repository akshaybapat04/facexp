function [new_bel, niter] = bp_mrf2(adj_mat, pot, local_evidence, varargin)
% BP_MRF2 Belief propagation on an MRF with pairwise potentials
% function [bel, niter] = bp_mrf2(adj_mat, pot, local_evidence, varargin)
%
% Input:
% adj_mat(i,j) = 1 iff there is an edge between nodes i and j
% pot(kj,ki,i,j) or pot{i,j}(kj,ki) = potential on edge between nodes i,j
%   If the potentials on all edges are the same,
%   you can just pass in 1 array, pot(kj,ki)
% local_evidence(k,i) or local_evidence{i}(k) = Pr(observation at node i | Xi=k)
%
% Use cell arrays if the hidden nodes do not all have the same number of values.
%
% Output:
% bel(k,i) or bel{i}(k) = P(Xi=k|evidence)
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


nnodes = length(adj_mat);

[max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

if iscell(local_evidence)
  use_cell = 1;
  tied_pot = 0;
  nnodes = length(local_evidence);
else
  use_cell = 0;
  tied_pot = (ndims(pot)==2);
  [nstates nnodes] = size(local_evidence);
end


% give each edge a unique number
ndx = find(adj_mat);
nedges = length(ndx);
edge_id = zeros(1, nnodes*nnodes);
edge_id(ndx) = 1:nedges; 
edge_id = reshape(edge_id, nnodes, nnodes);

% initialise messages
if use_cell
  prod_of_msgs = cell(1, nnodes);
  old_bel = cell(1, nnodes);
  nstates = zeros(1, nnodes);
  old_msg = cell(1, nedges);
  for i=1:nnodes
    nstates(i) = length(local_evidence{i});
    prod_of_msgs{i} = local_evidence{i};
    old_bel{i} = local_evidence{i};
  end
  for i=1:nnodes
    nbrs = find(adj_mat(:,i));
    for j=nbrs(:)'
      old_msg{edge_id(i,j)} = normalise(ones(nstates(j),1));
    end
  end
else
  prod_of_msgs = local_evidence;
  old_bel = local_evidence;
  %old_msg = zeros(nstates, nnodes, nnodes); 
  old_msg = zeros(nstates, nedges); 
  m = normalise(ones(nstates,1));
  for i=1:nnodes
    nbrs = find(adj_mat(:,i));
    for j=nbrs(:)'
      old_msg(:, edge_id(i,j)) = m;
      %old_msg(:,i,j) = m;
    end
  end
end


converged = 0;
iter = 1;

while ~converged & (iter <= max_iter)

  % each node sends a msg to each of its neighbors
  for i=1:nnodes
    nbrs = find(adj_mat(i,:));
    for j=nbrs(:)'
      if tied_pot
	pot_ij = pot;
      else
	if use_cell
	  pot_ij = pot{i,j};
	else
	  pot_ij = pot(:,:,i,j);
	end
      end
      % Compute temp = product of all incoming msgs except from j
      % by dividing out old msg from j from the product of all msgs sent to i
      if use_cell
	temp = prod_of_msgs{i};
	m = old_msg{edge_id(j,i)};
      else
	temp = prod_of_msgs(:,i);
	m = old_msg(:, edge_id(j,i));
      end
      m = m + (m==0); % valid since m(k)=0 => temp(k)=0, so can replace 0's with anything
      temp = temp ./ m;
      if maximize
	newm = max_mult(pot_ij, temp); % bottleneck
      else
	newm = pot_ij * temp;
      end
      newm = normalise(newm);
      if use_cell
	new_msg{edge_id(i,j)} = newm;
      else
	new_msg(:, edge_id(i,j)) = newm;
      end
    end % for j 
  end % for i

  % each node multiplies all its incoming msgs and computes its local belief
  if use_cell
    for i=1:nnodes
      nbrs = find(adj_mat(:,i));
      prod_of_msgs{i} = local_evidence{i};
      for j=nbrs(:)'
	prod_of_msgs{i} = prod_of_msgs{i} .* new_msg{edge_id(j,i)};
      end
      new_bel{i} = normalise(prod_of_msgs{i});
    end
    err = abs(cat(1,new_bel{:}) - cat(1, old_bel{:}));
  else
    for i=1:nnodes
      nbrs = find(adj_mat(:,i));
      prod_of_msgs(:,i) = local_evidence(:,i);
      for j=nbrs(:)'
	prod_of_msgs(:,i) = prod_of_msgs(:,i) .* new_msg(:,edge_id(j,i));
      end
      new_bel(:,i) = normalise(prod_of_msgs(:,i));
    end
    err = abs(new_bel(:) - old_bel(:));
  end
  converged = all(err < tol);
  if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
  iter = iter + 1;
  old_msg = new_msg;
  old_bel = new_bel;
end % while

niter = iter-1;

fprintf('converged in %d iterations\n', niter);

