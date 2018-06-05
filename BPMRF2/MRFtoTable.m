function [T, Z, ass] = MRFtoTable(E, pot, nstates, local_evidence)
% COMPUTE_JOINT_FROM_MRF2 Compute large table encoding joint probability of all nodes
%
% function [T, Z, ass] = MRFtoTable(E, pot, nstates, local_evidence)
% local_evidence{i}(Xi)
% Output
% T(a) = prob of a'th joint assignment
% Z = partiton fn
% ass(a,:) is the a'th assignment

if nargin < 4, local_evidence = []; end

nnodes = length(E);
Nass = prod(nstates);
ass = ind2subv(nstates, 1:Nass);
T = zeros(1,Nass);
for a=1:size(ass,1)
  vals = ass(a,:);
  p=1;
  for i=1:nnodes
    if ~isempty(local_evidence) & ~isempty(local_evidence{i})
      p = p * local_evidence{i}(vals(i));
    end
  end
  for i=1:nnodes
    for j=i:nnodes
      e = E(i,j);
      if E(i,j)>0
	p = p * pot{e}(vals(i), vals(j));
      end
    end
  end
  T(a) = p;
end
[T, Z] = normalise(T);


