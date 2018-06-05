function [new_bel, niter, in_msgs, prod_of_msgs] = bp_mrf2_vectorized(pot, local_evidence, varargin)
% BP_MRF2 Belief propagation on an MRF with pairwise potentials
% function [bel, niter] = bp_mrf2_lattice(pot, local_evidence, varargin)
%
% Input:
% pot(k1,k2) = potential on edge between nodes i,j (must be the same for all pairs)
% local_evidence(r,c,j) = Pr(observation at node (r,c) | Xi=j); 3D matrix
%
% Output:
% bel(r,c,k) = P(X(r,c)=k|evidence)
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

[nrows ncols nstates] = size(local_evidence);
nnodes=nrows*ncols;

[max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-2, 'maximize', 0, 'verbose', 0);

r=[2:nrows+1]; %rows we care about in message matrices (ignore edges)
c=[2:ncols+1]; %columns we care about

% initialise messages
prod_of_msgs = ones(nrows+2,ncols+2,nstates);
prod_of_msgs(r,c,:)=local_evidence;
old_bel = local_evidence;
new_bel=old_bel;

% we want all ones along edges, so multiplications don't affect interior
old_msgs = ones(nrows+2, ncols+2, 4, nstates);
old_msgs(r, c, :, :) = normalize(ones(nrows,ncols,4,nstates),4);
msgs = old_msgs;

converged = 0;
iter = 1;


 
while ~converged & (iter <= max_iter)
  
  oldMsgsRaw = msgs;
  old_msgs=msgs+(msgs==0); %get rid of zero terms, since we'll be dividing

  %  x1 - x2 - x3
      % Consider 2 nodes in the same row but neighboring columns (c-1) - c.
    % The msg coming from the west into c, m(r,c,west,i), is given by
    % sum_{j} pot(i,j) * tmp(j),
    % where tmp(j) is the product of all msgs coming into (c-1) except
    % those coming from c (from the east), i.e., 
    % tmp = prod_of_msgs(r,c-1,:) / oldm(r,c-1,east,:).
    % For nodes on the edge, tmp will contain a uniform message from the dummy direction.

  % msgs(r,c,1,:) is msg (r,c) gets from east
  % msgs(r,c,3,:) is msg (r,c) gest from west
  % msgs(r,c,2,:) is msg (r,c) gets from south
  % msgs(r,c,4,:) is msg (r,c) gets from north
  msgs(r,c,1,:)=prod_of_msgs(r,c-1,:)./permute(old_msgs(r,c-1,3,:), [1 2 4 3]);
  msgs(r,c,3,:)=prod_of_msgs(r,c+1,:)./permute(old_msgs(r,c+1,1,:), [1 2 4 3]);
  msgs(r,c,2,:)=prod_of_msgs(r+1,c,:)./permute(old_msgs(r+1,c,4,:), [1 2 4 3]);
  msgs(r,c,4,:)=prod_of_msgs(r-1,c,:)./permute(old_msgs(r-1,c,2,:), [1 2 4 3]);
  
  %now multiply by the potential (tricky because msgs is 4D)
  R=reshape(msgs(r, c, :, :), nrows*ncols*4, nstates)';
  if maximize
    M=max_mult(pot, R);
  else
    M=pot*R;
  end
  M=reshape(M, nstates, nrows, ncols, 4);
  msgs(r,c,:,:)=permute(M, [2 3 4 1]);

  %normalize msgs to prevent overflow
  msgs(r,c,:,:)=normalize(msgs(r,c,:,:), 4);
  

  % The edges absorb pot*ones(nstates,1) from the boundaries.
  % If each row of the potential has a different sum, this introduces errors.
  % Hence we set all the msgs coming in from edges to 1s.
  msgs(:,ncols+1,3,:) = 1; % RHS absorbs from east
  msgs(:,2, 1,:) = 1; % LSH absorbs from west
  msgs(nrows+1,:,2,:) = 1; % bottom absorbs from souht
  msgs(2,:,4,:) = 1; % top absorbs from north

  new_msgs = msgs;
  msgs = momentum*oldMsgsRaw + (1-momentum)*new_msgs;
  
  %product is just product of all 4 and local evidence (leaving ones on edges)
  %permute removes singleton dimension 3 (faster than squeeze)
  prod_of_msgs(r,c,:)=permute(prod(msgs(r,c,:,:), 3), [1 2 4 3]).*local_evidence(r-1,c-1,:);

  
  old_bel=new_bel;
  new_bel=normalize(prod_of_msgs(r,c,:), 3);
  err = abs(new_bel-old_bel);
  converged = all(all(all(err < tol)));
  if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
  iter = iter + 1;
end % while

niter = iter-1;

if ~converged, fprintf('warning: BP did not converge in %d iterations\n', niter); end

if verbose, fprintf('converged in %d iterations\n', niter); end

in_msgs = msgs(2:end-1, 2:end-1, :, :);
