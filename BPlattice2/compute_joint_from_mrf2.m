function [T, Z, ass] = compute_joint_from_mrf2(adj_mat, pot, nstates, local_evidence)
% COMPUTE_JOINT_FROM_MRF2 Compute large table encoding joint probability of all nodes
% function [T, Z, ass] = compute_joint_from_mrf2(adj_mat, pot, nstates, local_evidence)
%
% Input:
% adj_mat(i,j) = 1 iff there is an edge between nodes i and j
% pot{e}(ki,kj) on edge e  % pot{i,j}(kj,ki) = potential on edge between nodes i,j
% nstates(i) = num values for node i
% local_evidence{i}(j) = Pr(observation at node i | Xi=j) (optional)
%
% Output
% T(a) = prob of a'th joint assignment
% Z = partiton fn
% ass(a,:) is the a'th assignment

if nargin < 4, local_evidence = []; end

[E, Nedges] = assignEdgeNums(adj_mat);
nnodes = length(adj_mat);
Nass = prod(nstates);
ass = ind2subv(nstates, 1:Nass);
T = zeros(1,Nass);
for a=1:size(ass,1)
  vals = ass(a,:);
  p=1;
  if ~isempty(local_evidence)
    for i=1:nnodes
      p = p * local_evidence{i}(vals(i));
    end
  end
  for i=1:nnodes
    for j=i:nnodes
      if adj_mat(i,j)
	%p = p * pot{i,j}(vals(j), vals(i));
	%p = p * pot{i,j}(vals(i), vals(j));
	e = E(i,j);
	p = p * pot{e}(vals(i), vals(j));
      end
    end
  end
  T(a) = p;
end
[T, Z] = normalise(T);


