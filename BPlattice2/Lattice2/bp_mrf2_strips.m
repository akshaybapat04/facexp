function [new_bel, niter, msgs] = bp_mrf2_strips(pot, local_evidence, varargin)
% BP_MRF2 Belief propagation on an MRF with pairwise potentials
% function [bel, niter] = bp_mrf2_lattice(pot, local_evidence, varargin)
%
% This is a modified version of bp_mrf2_lattice_vectorized.
% where we iterate over vertical strips (sets of neighboring columns).
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
% nstrips - number of parallel columns to divide image into for simultaneous processing [1]
% max_iter - max. num. iterations [ 5*nnodes]
% momentum - weight assigned to old message in convex combination
%            (useful for damping oscillations) - currently ignored [0]
% tol      - tolerance used to assess convergence [1e-3]
% maximize - 1 means use max-product, 0 means use sum-product [0]
% verbose - 1 means print error at every iteration [0]

%s = sum(pot,2);
%%if max(s-s(1))>1-e3
%  error('kernel must have constant row sum')
%end


[nrows ncols nstates] = size(local_evidence);
nnodes=nrows*ncols;

[nstrips, max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'nstrips', 1, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

ncols_per_strip = ncols/nstrips;
if mod(ncols, nstrips)>0
  error(sprintf('nstrips %d must be a multiple of ncols %d', nstrips, ncols))
end


% We add dummy rows and columns on the boudnaries which always send msgs of all 1s
% and which never get updated. This simplifies the code.

rows = 2:nrows+1; % skip row 1, which is dummy
cols = 2:ncols+1; % skip col 1, which is dummy 

% initialise messages
prod_of_msgs = ones(nrows+2,ncols+2,nstates);
prod_of_msgs(rows,cols,:)=local_evidence;
old_bel = local_evidence;
new_bel=old_bel;

% msgs(r,c,dir,:) is the message sent *into* (r,c) from direction dir
% where dir=1 comes from west, 3 from east, 2 from south, 4 from north.
% We want all ones along edges, so multiplications don't affect interior.
old_msgs = ones(nrows+2, ncols+2, 4, nstates);
old_msgs(rows, cols, :, :) = normalize(ones(nrows,ncols,4,nstates),4);
msgs = old_msgs;

% Storage requirements
% The following arrays have size nrows*ncols*4*nstates: new_bel, old_bel
% % The following arrays have size (nrows+2)*(ncols+2)*4*nstates: msgs, old_msgs, prod_of_msgs

converged = 0;
iter = 1;

while ~converged & (iter <= max_iter)

  % copy msgs in interior into old_msgs before modifying 
  for s=1:nstrips
    cols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip)+1;
    %get rid of zero terms, since we'll be dividing
    old_msgs(rows,cols,:,:) = msgs(rows,cols,:,:) + (msgs(rows,cols,:,:) ==0);
  end
  
  % compute new msgs  in interior (may read old_msgs on boundary)
  for s=1:nstrips
    cols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip)+1;
    
    
    tmp = zeros(nrows, ncols_per_strip, 4, nstates);
    tmp(:,:,1,:)=prod_of_msgs(rows,cols-1,:)./permute(old_msgs(rows,cols-1,3,:), [1 2 4 3]);
    tmp(:,:,3,:)=prod_of_msgs(rows,cols+1,:)./permute(old_msgs(rows,cols+1,1,:), [1 2 4 3]);
    tmp(:,:,2,:)=prod_of_msgs(rows+1,cols,:)./permute(old_msgs(rows+1,cols,4,:), [1 2 4 3]);
    tmp(:,:,4,:)=prod_of_msgs(rows-1,cols,:)./permute(old_msgs(rows-1,cols,2,:), [1 2 4 3]);
    
    %now multiply by the potential (tricky because msgs is 4D)
    R=reshape(tmp, nrows*ncols_per_strip*4, nstates)';
    if maximize
      M=max_mult(pot, R);
    else
      M=pot*R;
    end
    M=reshape(M, [nstates nrows ncols_per_strip 4]);
    msgs(rows,cols,:,:)=permute(M, [2 3 4 1]);

    %normalize msgs to prevent underflow
    msgs(rows,cols,:,:)=normalize(msgs(rows,cols,:,:), 4);
  end
  
  % The edges absorb pot*ones(nstates,1) from the boundaries.
  % If each row of the potential has a different sum, this introduces errors.
  % Hence we set all the msgs coming in from edges to 1s.
  msgs(:,ncols+1,3,:) = 1; % RHS absorbs from east
  msgs(:,2, 1,:) = 1; % LSH absorbs from west
  msgs(nrows+1,:,2,:) = 1; % bottom absorbs from souht
  msgs(2,:,4,:) = 1; % top absorbs from north

  
  % Compute local beliefs (excluding boundaries)
  for s=1:nstrips
    cols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip)+1;

    % Take product of all incoming messages along all directions (dimension 3).
    % The result has size nrows*ncols*1*nstates.
    % Permute removes singleton dimension 3 (faster than squeeze).
    % We do not update prod_of_msgs for the dummy edges.
    prod_of_msgs(rows,cols,:)=permute(prod(msgs(rows,cols,:,:), 3), [1 2 4 3]).*local_evidence(rows-1,cols-1,:);

    % Compute beliefs for cols=1:ncols (exclude dummy boundaries)
    old_bel(:,cols-1,:) = new_bel(:,cols-1,:);
    new_bel(:,cols-1,:) = normalize(prod_of_msgs(rows, cols,:), 3);
    diff = (new_bel(:,cols-1,:)-old_bel(:,cols-1,:));
    err(s) = max(diff(:));
  end
  converged = max(err)<tol;
  if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
  iter = iter + 1;
end % while

niter = iter-1;

fprintf('converged in %d iterations\n', niter);
