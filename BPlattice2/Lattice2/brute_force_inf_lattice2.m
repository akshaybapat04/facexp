function [bel, bel2, negloglik, mpe] = brute_force_inf_lattice2(kernel, local_evidence)
% function [bel, bel2, negloglik] = brute_force_inf_lattice2(kernel, local_evidence)
% Input:
% kernel(k1,k2) = potential on edge between nodes i,j (must be the same for all pairs)
% local_evidence(r,c,j) = Pr(observation at node (r,c) | Xi=j); 3D matrix
%
% Output:
% bel(r,c,k) = P(X(r,c)=k|evidence)
% bel2(q1,q2,e)
% negative log likelihood


[nrows ncols nstates] = size(local_evidence);
[T, Z, ass] = compute_joint_from_lattice2(kernel, local_evidence);
negloglik = -log(Z);
mpe = reshape(ass(argmax(T), :), nrows, ncols);

nnodes = nrows*ncols;
[coords, edge] = edge_num_lattice2(nrows, ncols);
Nedges = (nrows-1)*ncols + nrows*(ncols-1);

bel = zeros(nrows, ncols, nstates);
bel2 = zeros(nstates, nstates, Nedges);

for r=1:nrows
  for c=1:ncols
    i = sub2ind([nrows ncols], r, c);
    b = marginalize_table(T, 1:nnodes, nstates*ones(1,nnodes), i);
    bel(r,c,:) = b(:);
    
    if c<ncols
      j = sub2ind([nrows ncols], r, c+1);
      e = edge(r,c, r,c+1);
      b2 = marginalize_table(T, 1:nnodes, nstates*ones(1,nnodes), [i j]);
      bel2(:,:,e) = b2;
    end
    if r<nrows
      j = sub2ind([nrows ncols], r+1, c);
      e = edge(r,c, r+1,c);
      b2 = marginalize_table(T, 1:nnodes, nstates*ones(1,nnodes), [i j]);
      bel2(:,:,e) = b2;
    end
  end
end
