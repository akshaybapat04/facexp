function [T, Z, ass] = compute_joint_from_lattice2(kernel, local_evidence)
% COMPUTE_JOINT_FROM_LATTICE2 Compute large table encoding joint probability of all nodes
% function [T, Z, ass] = compute_joint_from_mrf2(adj_mat, pot, nstates, local_evidence)
%
% Input:
% kernel(q,q')
% local_evidence(r,c,k)
%
% Output
% T(a) = prob of a'th joint assignment
% Z = partiton fn
% ass(a,:) is the a'th assignment

[nrows ncols nstates] = size(local_evidence);
nnodes = nrows*ncols;
Nass = nstates^nnodes;
ass = ind2subv(nstates*ones(1,nnodes), 1:Nass);
T = zeros(1,Nass);
for a=1:size(ass,1)
  vals = reshape(ass(a,:), nrows, ncols);
  p=1;
  for r=1:nrows
    for c=1:ncols
      p = p * local_evidence(r,c,vals(r,c));
      if c<ncols
	p = p * kernel(vals(r,c), vals(r,c+1));
      end
      if r<nrows
	p = p * kernel(vals(r,c), vals(r+1,c));
      end
    end
  end
  T(a) = p;
end
[T, Z] = normalise(T);


