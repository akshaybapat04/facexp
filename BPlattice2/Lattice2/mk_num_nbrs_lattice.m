function nnbrs = mk_nnbrs_lattice(nrows, ncols)
% function nnbrs = mk_nnbrs_lattice(nrows, ncols)
% nnbrs(r,c) = num of neighbors of node (r,c) in a 2D lattice
% assuming 4 nearest nbr connected with no wrap around
%
% e.g., for 4x4:
%     2     3     3     2
%     3     4     4     3
%     3     4     4     3
%     2     3     3     2


if nrows==1 
  nnbrs = 2*ones(1, ncols);
  nnbrs(1) = 1;
  nnbrs(end) = 1;
elseif ncols==1
  nnbrs = 2*ones(nrows, 1);
  nnbrs(1) = 1;
  nnbrs(end) = 1;
else  
  nnbrs = 4*ones(nrows, ncols);
  nnbrs(1,:) = 3;
  nnbrs(nrows,:) = 3;
  nnbrs(:,1) = 3;
  nnbrs(:,ncols) = 3;
  nnbrs(1,1) = 2; nnbrs(1,ncols) = 2;
  nnbrs(nrows,1) = 2; nnbrs(nrows,ncols) = 2;
end
