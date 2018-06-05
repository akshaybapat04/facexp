function [new_bel, niter] = bp_mrf2_debug(adj_mat, pot, local_evidence, varargin)
% BP_MRF2 Belief propagation on an MRF with pairwise potentials
% function [bel, niter] = bp_mrf2(adj_mat, pot, local_evidence, varargin)
%
% Input:
% adj_mat(i,j) = 1 iff there is an edge between nodes i and j
% pot{i,j}(kj,ki) = potential on edge between nodes i,j
%   If the potentials on all edges are the same,
%   you can just pass in 1 array, pot(kj,ki)
% local_evidence{i}(j) = Pr(observation at node i | Xi=j)
%
% Output:
% bel{i}(k) = P(Xi=k|evidence)
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
    process_options(varargin, 'max_iter', [], 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

if isempty(max_iter) % no user supplied value, so compute default
  max_iter = 5*nnodes;
  %if acyclic(adj_mat, 0) -- can be very slow!
  %  max_iter = nnodes;
  %else
  %  max_iter = 5*nnodes;
  %end
end

if iscell(pot)
  tied_pot = 0;
else
  tied_pot = 1;
end

% initialise messages
prod_of_msgs = cell(1, nnodes);
old_bel = cell(1, nnodes);
nstates = zeros(1, nnodes);
old_msg = cell(nnodes, nnodes);
for i=1:nnodes
  nstates(i) = length(local_evidence{i});
  prod_of_msgs{i} = local_evidence{i};
  old_bel{i} = local_evidence{i};
end
for i=1:nnodes
  nbrs = find(adj_mat(:,i));
  for j=nbrs(:)'
    old_msg{i,j} = normalise(ones(nstates(j),1));
  end
end

converged = 0;
iter = 1;

div_time = 0;
mult_time = 0;

while ~converged & (iter <= max_iter)

  % each node sends a msg to each of its neighbors
  for i=1:nnodes
    nbrs = find(adj_mat(i,:));
    for j=nbrs(:)'
      if tied_pot
	pot_ij = pot;
      else
	pot_ij = pot{i,j};
      end
      
      % Compute temp = product of all incoming msgs except from j
      
      % by dividing out old msg from j from the product of all msgs sent to i
      %tic;
      temp = prod_of_msgs{i};
      m = old_msg{j,i};
      m = m + (m==0); % valid since m(k)=0 => temp(k)=0, so can replace 0's with anything
      temp = temp ./ m;
      %t=toc;
      %div_time = div_time + t;

      if 0
      % Explicitly multiply all incoming msgs except from j
      % and check it gives same results as division.
      % This is faster if num nbrs except j <= C, since we do C multiplies
      % instead of 1 divide. (C is probably 1 or 2; for 4 neighbors, division
      % is twice as fast.)
      new_msg_div = normalise(pot_ij * temp);
      tic;
      mask = ((1:nnodes) ~= j);
      nbrs_exj = find(adj_mat(:,i) .* mask');
      temp = local_evidence{i};
      for k=nbrs_exj(:)'
	temp = temp .* old_msg{k,i};
      end
      t=toc;
      mult_time = mult_time + t;
      new_msg_mult = normalise(pot_ij * temp);
      assert(approxeq(new_msg_div, new_msg_mult))
      end
      
      if maximize
	new_msg{i,j} = max_mult(pot_ij, temp);
      else
	new_msg{i,j} = pot_ij * temp;
      end
      %new_msg{i,j} = normalise(new_msg{i,j});
      c = sum(new_msg{i,j}); d = c + (c==0); 
      new_msg{i,j} = new_msg{i,j} / d;
      
    end % for j 
  end % for i

  % each node multiplies all its incoming msgs and computes its local belief
  for i=1:nnodes
    nbrs = find(adj_mat(:,i));
    prod_of_msgs{i} = local_evidence{i};
    for j=nbrs(:)'
      prod_of_msgs{i} = prod_of_msgs{i} .* new_msg{j,i};
    end
    new_bel{i} = normalise(prod_of_msgs{i});
  end

  err = abs(cat(1,new_bel{:}) - cat(1, old_bel{:}));
  converged = all(err < tol);
  %converged = approxeq(cat(1,new_bel{:}),  cat(1, old_bel{:}), tol);
  if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
  iter = iter + 1;
  old_msg = new_msg;
  old_bel = new_bel;
end % while

niter = iter-1;

fprintf('converged in %d iterations\n', niter);
%div_time
%mult_time

%%%%%%%%

function y=max_mult(A,x)

% y(j) = max_i A(j,i) x(i)

%X=ones(size(A,1),1) * x(:)'; % X(j,i) = x(i)
%y=max(A.*X, [], 2);

% This is faster
X=x*ones(1,size(A,1));
y=max(A'.*X)';
