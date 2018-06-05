function [new_bel, niter, in_msg] = bp_mrf2_forloops(pot, local_evidence, varargin)
% BP_MRF2_LATTICE_SLOW Belief propagation on an MRF with pairwise potentials
% function [bel, niter] = bp_mrf2_lattice_slow(pot, local_evidence, varargin)
%
% Same as bp_mrf2_lattice, except it is not vectorized.

[nrows ncols nstates] = size(local_evidence);
nnodes = nrows*ncols;

[max_iter, momentum, tol, maximize, verbose] = ...
    process_options(varargin, 'max_iter', 5*nnodes, 'momentum', 0, ...
		    'tol', 1e-3, 'maximize', 0, 'verbose', 0);

[out_edge, in_edge, nedges] = assign_edge_numbers(nrows, ncols);

% initialise messages
prod_of_msgs = permute(local_evidence, [3 1 2]); % prod(q,r,c)
old_bel = local_evidence;
old_msg = zeros(nstates, nedges); 
m = normalise(ones(nstates,1));
for c=1:ncols
  for r=1:nrows
    for dir=1:4
      e = out_edge(r,c,dir);
      if e>0, old_msg(:,e) = m; end
    end % dir
  end % r
end % c

converged = 0;
iter = 1;
while ~converged & (iter <= max_iter)
  
  % each node sends a msg to each of its neighbors in each direction
  for c=1:ncols
    for r=1:nrows
      for dir=1:4
	out_e = out_edge(r,c,dir);
	in_e = in_edge(r,c,dir);
	if out_e>0 & in_e>0
	  %fprintf('r=%d,c=%d, send in dir%d\n', r,c,dir);
	  new_msg(:,out_e) = compute_msg(prod_of_msgs(:,r,c), old_msg(:,in_e), pot, maximize);
	end
      end % dir
    end % r
  end % c
    
  % each node multiplies all its incoming msgs and computes its local belief
  for c=1:ncols
    for r=1:nrows
      prod_of_msgs(:,r,c) = local_evidence(r,c,:);
      for dir=1:4
	e = in_edge(r,c,dir);
	if e > 0
	  prod_of_msgs(:,r,c) = prod_of_msgs(:,r,c) .* new_msg(:,e); 
	  in_msg(r,c,dir,:) = new_msg(:,e);
	end
      end
      new_bel(:,r,c) = normalise(prod_of_msgs(:,r,c));
    end
  end

  if 0
  disp('for loops')
  squeeze(in_msg(:,:,1,1))
  squeeze(in_msg(:,:,3,1))
  squeeze(permute(prod_of_msgs, [2 3 1]))
  keyboard
  end
  
  err = abs(new_bel(:) - old_bel(:));
  converged = all(err < tol);
  if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
  iter = iter + 1;
  old_msg = new_msg;
  old_bel = new_bel;
end % while

niter = iter-1;

fprintf('converged in %d iterations\n', niter);

new_bel = permute(new_bel, [2 3 1]); % now bel(r,c,k)

%%%%%%
function out_msg = compute_msg(prod_of_msg, in_msg, pot, maximize)

m = in_msg + (in_msg==0);
temp = prod_of_msg ./ m;
if maximize
  newm = max_mult(pot, temp); 
else
  newm = pot * temp;
end
out_msg = normalise(newm);



%%%%%%%%%%%%

  
function [out_edge, in_edge, nedges] = assign_edge_numbers(nrows, ncols)

% assign each edge in each direction a unique number
%north = 1; east = 2; south = 3; west = 4;
north = 2; east = 1; south = 4; west = 3;
ndir = 4; % num directions to send msgs in
out_edge = zeros(nrows, ncols, ndir);
in_edge = zeros(nrows, ncols, ndir);
e = 1;
% north
for r=2:nrows
  for c=1:ncols
    out_edge(r,c,north) = e;
    in_edge(r-1,c,south) = e;
    e = e+1;
  end
end
% east
for r=1:nrows
  for c=1:ncols-1
    out_edge(r,c,east) = e;
    in_edge(r,c+1,west) = e;
    e = e+1;
  end
end
% south
for r=1:nrows-1
  for c=1:ncols
    out_edge(r,c,south) = e;
    in_edge(r+1,c,north) = e;
    e = e+1;
  end
end
% west
for r=1:nrows
  for c=2:ncols
    out_edge(r,c,west) = e;
    in_edge(r,c-1,east) = e;
    e = e+1;
  end
end
nedges = e-1;
