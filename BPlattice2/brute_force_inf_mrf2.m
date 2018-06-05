function [bel, bel2, negloglik, mpe] = brute_force_inf_mrf2(adj_mat, pot, nstates, local_evidence)
% BRUTE_FORCE_INF_MRF2 Enumerate the full joint
% function [bel, bel2, negloglik, mpe] = brute_force_inf_mrf2(adj_mat, pot, nstates, local_evidence)
%
% Input:
% local_evidence{i}(j) = Pr(observation at node i | Xi=j)
% pot{i,j}(ki,kj) = potential on edge between nodes i,j
% or pot{e}(ki,kj)
% adj_mat(i,j) = 1 iff there is an edge between nodes i and j
%
% Output
% bel{i}(j) = P(Xi=j|evidence)
% bel2{i,j}(qi,qj) for j>=i
% mpe(i) = most probable value of node i

if ~iscell(local_evidence)
  local_evidence = num2cell(local_evidence, 1);
  pot = squeeze(num2cell(pot, [1 2]));
  use_cell = 0;
else
  use_cell = 1;
end
 
[T, Z, ass] = compute_joint_from_mrf2(adj_mat, pot, nstates, local_evidence);
negloglik = -log(Z+eps);

nnodes = length(adj_mat);
bel = cell(1,nnodes);
for i=1:nnodes
  for j=1:nstates(i)
    bel{i}(j) = sum(T(find(ass(:,i)==j)));
  end
  bel{i} = bel{i}(:);

  m = marginalize_table(T, 1:nnodes, nstates, i);
  assert(approxeq(m(:), bel{i}));

  for j=i:nnodes
    if adj_mat(i,j)
      dom = [i j];
      m2 = marginalize_table(T, 1:nnodes, nstates, dom);
      bel2{i,j} = m2;
    end
  end
end

mpe = ass(argmax(T), :);

if ~use_cell
  bel = cell2num(bel);
end


