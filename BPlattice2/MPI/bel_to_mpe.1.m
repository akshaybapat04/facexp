function mpe = bel_to_mpe(bel)
% BEL_TO_MPE Convert marginal beliefs to most probable explanation (Viterbi decoding)
% function mpe = bel_to_mpe(bel)
%
% We pick the locally maximal value.
% This may give inconsistent results if there are ties.
%
% bel(:,i) or bel{i}(:) is the distribution over states for node i

if iscell(bel)
  nnodes = length(bel);
  mpe = zeros(1,nnodes);
  for i=1:nnodes
    [m, mpe(i)] = max(bel{i});
    if length(find(bel{i}==m))>1
      fprintf('warning: node %d has ties for MPE\n', i);
    end
  end
else
  nnodes = size(bel,2);
  mpe = zeros(1,nnodes);
  for i=1:nnodes
    [m, mpe(i)] = max(bel(:,i));
    if length(find(bel(:,i)==m))>1
      fprintf('warning: node %d has ties for MPE\n', i);
    end
  end
end
