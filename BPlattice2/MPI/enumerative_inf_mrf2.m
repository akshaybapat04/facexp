function [bel, mpe] = enumerative_inf_mrf2(adj_mat, pot, local_evidence)
% ENUMERATIVE_INF_MRF2 Brute force computation of marginals of mrf2
% function [bel, mpe] = enumerative_inf_mrf2(adj_mat, pot, local_evidence)
%
% Input:
% local_evidence{i}(j) = Pr(observation at node i | Xi=j)
% pot{i,j}(k1,k2) = potential on edge between nodes i,j
% adj_mat(i,j) = 1 iff there is an edge between nodes i and j
%
% Output
% bel{i}(j) = P(Xi=j|evidence)
% mpe(i) = most probable value of node i

[T, ass] = compute_global_joint_from_mrf2(local_evidence, pot, adj_mat);
nnodes = length(adj_mat);
for i=1:nnodes
  nstates(i) = length(local_evidence{i});
end
bel = cell(1,nnodes);
for i=1:nnodes
  for j=1:nstates(i)
    bel{i}(j) = sum(T(find(ass(:,i)==j)));
  end
  bel{i} = bel{i}(:);
end
mpe = ass(argmax(T), :);

%%%%%%%%%

function [T, ass] = compute_global_joint_from_mrf2(local_evidence, pot, adj_mat)
% COMPUTE_GLOBAL_JOINT_FROM_MRF2 Compute large table encoding joint probability of all nodes
% function [T, ass] = compute_global_joint_from_mrf2(local_evidence, pot, adj_mat)
%
% Input:
% local_evidence{i}(j) = Pr(observation at node i | Xi=j)
% pot{i,j}(kj,ki) = potential on edge between nodes i,j
% adj_mat(i,j) = 1 iff there is an edge between nodes i and j
%
% Output
% T(a) = prob of a'th joint assignment
% ass(a,:) is the a'th assignemtn

nnodes = length(adj_mat);
for i=1:nnodes
  nstates(i) = length(local_evidence{i});
end
N = prod(nstates);
ass = ind2subv(nstates, 1:N);
T = zeros(1,N);
for a=1:size(ass,1)
  vals = ass(a,:);
  p=1;
  for i=1:nnodes
    p = p * local_evidence{i}(vals(i));
  end
  for i=1:nnodes
    for j=i:nnodes
      if adj_mat(i,j)
	p = p * pot{i,j}(vals(j), vals(i));
      end
    end
  end
  T(a) = p;
end
T = normalise(T);

