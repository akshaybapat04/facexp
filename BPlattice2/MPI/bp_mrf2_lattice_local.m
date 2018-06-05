function [bel, niter] = bp_mrf2_lattice_local(pot, local_evidence, varargin)
% BP_MRF2 Belief propagation on an MRF with pairwise potentials
% function [bel, niter] = bp_mrf2_lattice(pot, local_evidence, varargin)
%
% This is a modified version of bp_mrf2_lattice_strips,
% designed to be easy to parallelize.
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


[nrows ncols nstates] = size(local_evidence);

nnodes=nrows*ncols;
ndir = 4;

[nstrips, max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'nstrips', 1, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

ncols_per_strip = ncols/nstrips;
if mod(ncols, nstrips)>0
  error(sprintf('nstrips %d must be a multiple of ncols %d', nstrips, ncols))
end

local_ev = zeros(nrows, ncols_per_strip, nstates, nstrips);
for s=1:nstrips
  scols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip);
  local_ev(:,:,:,s) = local_evidence(:, scols, :);
end

% initialise local beliefs
prod_of_msgs = local_ev;
old_bel = local_ev;
new_bel = old_bel;

% msgs(r,c,dir,:,s) is the message sent *into* (r,c) from direction dir for strip s
% where dir=1 comes from west, 3 from east, 2 from south, 4 from north.
old_msgs = normalize(ones(nrows, ncols_per_strip, ndir, nstates, nstrips), 4);
msgs = old_msgs;

% We add dummy rows and columns on the boudnaries which always send msgs of all 1s
% and which never get updated. This simplifies the code.

% Messages on dummy boundaries are all 1s so as not to effect multiplications.
% Since we divide the interior into vertical strips, 
% the internal vertical boundaries will later be changed.
% The horizontal boundary messages are always clamped at 1, and do not need to be stored.
old_msgs_east = ones(nrows, ndir, nstates, nstrips);
old_msgs_west = ones(nrows, ndir, nstates, nstrips);
prod_of_msgs_east = ones(nrows, nstates, nstrips);
prod_of_msgs_west = ones(nrows, nstates, nstrips);

converged = 0;
iter = 1;

% while ~converged & (iter <= max_iter)
for iter=1:2

  % copy msgs in interior into old_msgs before modifying 
  for s=1:nstrips
    %get rid of zero terms, since we'll be dividing
    old_msgs(:,:,:,:,s) = msgs(:,:,:,:,s) + (msgs(:,:,:,:,s) ==0);
  end
  
  % receive from your neighbors
  for s=1:nstrips
    if s>1, old_msgs_east(:,:,:,s) = old_msgs(:,end,:,:,s-1); end
    if s<nstrips, old_msgs_west(:,:,:,s) = old_msgs(:,1,:,:,s+1); end
    if s>1, prod_of_msgs_east(:,:,s) = prod_of_msgs(:,end,:,s-1); end
    if s<nstrips, prod_of_msgs_west(:,:,s) = prod_of_msgs(:,1,:,s+1); end
  end

%  if (iter==1)
%    save serial
%  end

  % compute new msgs
  for s=1:nstrips
    msgs(:,:,:,:,s) = comp_msgs(old_msgs(:,:,:,:,s), prod_of_msgs(:,:,:,s), ...
				    old_msgs_east(:,:,:,s), old_msgs_west(:,:,:,s),...
				    prod_of_msgs_east(:,:,s), prod_of_msgs_west(:,:,s),...
				    pot, maximize);

  end
  
  % Compute local beliefs 
  for s=1:nstrips
    % Take product of all incoming messages along all directions (dimension 3).
    % The result has size nrows*ncols*1*nstates*nstrips
    % Permute removes singleton dimension 3 (faster than squeeze).
    prod_of_msgs(:,:,:,s)=permute(prod(msgs(:,:,:,:,s), 3), [1 2 4 3]);
    prod_of_msgs(:,:,:,s)=prod_of_msgs(:,:,:,s) .* local_ev(:,:,:,s);

    old_bel(:,:,:,s) = new_bel(:,:,:,s);
    new_bel(:,:,:,s) = normalize(prod_of_msgs(:,:,:,s), 3);
    diff = (new_bel(:,:,:,s)-old_bel(:,:,:,s));
    err(s) = max(diff(:));
  end
  
  converged = max(err)<tol;
  if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
%  iter = iter + 1;

end % while

niter = iter-1;

fprintf('converged in %d iterations\n', niter);

bel = zeros(nrows, ncols, nstates);
for s=1:nstrips
  scols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip);
  bel(:,scols,:) = new_bel(:,:,:,s);
end


%%%%%%%%%%%%

function msgs = comp_msgs(old_msgs_center, prod_of_msgs_center, ...
			  old_msgs_east, old_msgs_west, ...
			  prod_of_msgs_east, prod_of_msgs_west,...
			  pot, maximize)

[nrows ncols ndir nstates] = size(old_msgs_center);
msgs = ones(nrows, ncols, ndir, nstates);
rows = 2:nrows+1;
cols = 2:ncols+1;

old_msgs = ones(nrows+2, ncols+2, ndir, nstates);
old_msgs(rows,cols,:,:) = old_msgs_center;
old_msgs(rows,cols(1)-1,:,:) = old_msgs_east;
old_msgs(rows,cols(end)+1,:,:) = old_msgs_west;

prod_of_msgs = ones(nrows+2, ncols+2, nstates);
prod_of_msgs(rows,cols,:) = prod_of_msgs_center;
prod_of_msgs(rows,cols(1)-1,:) = prod_of_msgs_east;
prod_of_msgs(rows,cols(end)+1,:) = prod_of_msgs_west;

% Consider 2 nodes in the same row but neighboring columns (c-1) - c.
% The msg coming from the west into c, m(r,c,west,i), is given by
% sum_{j} pot(i,j) * tmp(j),
% where tmp(j) is the product of all msgs coming into (c-1) except
% those coming from c (from the east), i.e., 
% tmp = prod_of_msgs(r,c-1,:) / oldm(r,c-1,east,:).
% For nodes on the edge, tmp will contain a uniform message from the dummy direction.

tmp = zeros(nrows, ncols, ndir, nstates);
tmp(:,:,1,:)=prod_of_msgs(rows,cols-1,:)./permute(old_msgs(rows,cols-1,3,:), [1 2 4 3]);
tmp(:,:,3,:)=prod_of_msgs(rows,cols+1,:)./permute(old_msgs(rows,cols+1,1,:), [1 2 4 3]);
tmp(:,:,2,:)=prod_of_msgs(rows+1,cols,:)./permute(old_msgs(rows+1,cols,4,:), [1 2 4 3]);
tmp(:,:,4,:)=prod_of_msgs(rows-1,cols,:)./permute(old_msgs(rows-1,cols,2,:), [1 2 4 3]);

%now multiply by the potential (tricky because msgs is 4D)
R=reshape(tmp, nrows*ncols*ndir, nstates)';
if maximize
  M=max_mult(pot, R);
else
  M=pot*R;
end
M=reshape(M, [nstates nrows ncols ndir]);
msgs=permute(M, [2 3 4 1]);

%normalize msgs to prevent underflow
msgs=normalize(msgs, 4);
