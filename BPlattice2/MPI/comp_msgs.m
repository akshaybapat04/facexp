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
