function [bel, bel2, negloglik] = brute_force_inf_lattice2_wrapper(kernel, local_evidence)
% function [bel, bel2, negloglik] = brute_force_inf_lattice2(kernel, local_evidence)
% Input:
% kernel(k1,k2) = potential on edge between nodes i,j (must be the same for all pairs)
% local_evidence(r,c,j) = Pr(observation at node (r,c) | Xi=j); 3D matrix
%
% Output:
% bel(r,c,k) = P(X(r,c)=k|evidence)
% bel2(q1,q2,e)
% negative log likelihood

% This is just a wrapper for brute_force_inf_mrf2

[nrows ncols nstates] = size(local_evidence);
adj_mat = mk_2D_lattice(nrows, ncols);
nnodes = nrows*ncols;
pot = cell(nnodes, nnodes);
for i=1:nnodes
  for j=1:nnodes
    if adj_mat(i,j)
      pot{i,j} = kernel;
    end
  end
end

local_evidence_cell = num2cell(reshape(local_evidence, nrows*ncols, nstates)', 1);
[bel_cell, bel2_cell, negloglik, mpe] = brute_force_inf_mrf2(adj_mat, pot, nstates*ones(1,nrows*ncols), ...
				       local_evidence_cell);
mpe = reshape(mpe, [nrows ncols]);
i=1;
for c=1:ncols
  for r=1:nrows
    bel(r,c,:) = bel_cell{i};
    i = i+1;
  end
end

[coords, edge] = edge_num_lattice2(nrows, ncols);
Nedges = (nrows-1)*ncols + nrows*(ncols-1);
for e=1:Nedges
  r1 = coords(e,1); c1 = coords(e,2);  r2 = coords(e,3); c2 = coords(e,4);
  %[r1,c1,r2,c2] = deal(coords(e,:));
  i = sub2ind([nrows ncols], r1, c1);
  j = sub2ind([nrows ncols], r2, c2);
  bel2(:,:,e) = bel2_cell{i,j};
end


