function [new_bel, niter] = bp_mrf2_lattice(pot, local_evidence, varargin)
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

[rows cols nstates] = size(local_evidence);
nnodes=rows*cols;

[max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

r=[2:rows+1]; %rows we care about in message matrices (ignore edges)
c=[2:cols+1]; %columns we care about

% initialise messages
prod_of_msgs = ones(rows+2,cols+2,nstates);
prod_of_msgs(r,c,:)=local_evidence;
old_bel = local_evidence;
new_bel=old_bel;

% we want all ones along edges, so multiplications don't affect interior
old_msgs = ones(rows+2, cols+2, 4, nstates);
old_msgs(r, c, :, :) = normalize(ones(rows,cols,4,nstates),4);
msgs = old_msgs;

converged = 0;
iter = 1;


 
while ~converged & (iter <= max_iter)

%msgs, old_msgs will be (rows+2) x (cols+2) x states x 4  (1 msg for each dir)
%and messages coming in from boundaries

%bels will be rows x cols x nstates

  old_msgs=msgs+(msgs==0); %get rid of zero terms, since we'll be dividing

%msgs(:,:,:,1) come from north, 2 from east, 3 from south, 4 from west

  %message coming *in* from north is prod of ones that went into north node 
  %divided by that node's message coming in from south

  %permute is faster than squeeze
  msgs(r,c,1,:)=prod_of_msgs(r,c-1,:)./permute(old_msgs(r,c-1,3,:), [1 2 4 3]);

  %similarly for other dirs
  msgs(r,c,3,:)=prod_of_msgs(r,c+1,:)./permute(old_msgs(r,c+1,1,:), [1 2 4 3]);
  msgs(r,c,2,:)=prod_of_msgs(r+1,c,:)./permute(old_msgs(r+1,c,4,:), [1 2 4 3]);
  msgs(r,c,4,:)=prod_of_msgs(r-1,c,:)./permute(old_msgs(r-1,c,2,:), [1 2 4 3]);

  %now multiply by the potential (tricky because msgs is 4D)
  R=reshape(msgs(r, c, :, :), rows*cols*4, nstates)';
  if maximize
    M=max_mult(pot, R);
  else
    M=pot*R;
  end
  N=reshape(M, nstates, rows, cols, 4);
  msgs(r,c,:,:)=permute(N, [2 3 4 1]);

  %normalize msgs to prevent overflow
  msgs(r,c,:,:)=normalize(msgs(r,c,:,:), 4);
  
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

fprintf('converged in %d iterations\n', niter);
