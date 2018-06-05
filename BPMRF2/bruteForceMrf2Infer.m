function [bel, belE, logZ, mpe, T] = bruteForceMrf2Infer(engine, pot, localEv)
% BRUTE_FORCE_INF_MRF2 Enumerate the full joint
%
% Input:
% localEv{i}(j) = Pr(observation at node i | Xi=j)
% pot{e}(ki,kj) = potential on edge e
% E(i,j) = edge number
%
% Output
% bel{i}(j) = P(Xi=j|evidence)
% belE{e}(q1,q2)
% mpe(i) = most probable value of node i 

if nargin < 3, localEv = []; end

nstates = engine.nstates;
E = engine.E;
nnodes = length(E);
 
[T, Z, ass] = MRFtoTable(E, pot, nstates, localEv);
logZ = log(Z);

bel = cell(1,nnodes);
nedges = length(find(E(:)>0))/2;
belE = cell(1, nedges);
for i=1:nnodes
  bel{i} = marginalize_table(T, 1:nnodes, nstates, i);
  bel{i} = bel{i}(:);
  for j=i+1:nnodes
    if E(i,j)>0
      e = E(i,j);
      dom = [i j];
      m2 = marginalize_table(T, 1:nnodes, nstates, dom);
      belE{e} = reshape(m2, nstates(i), nstates(j));
    end
  end
end

mpe = ass(argmax(T), :);



