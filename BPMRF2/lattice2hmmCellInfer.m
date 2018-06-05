function [bel, belE, logZ] = lattice2hmmCellInfer(engine, pot, localEv, varargin)
% function [bel, belE, logZ] = lattice2hmmCellInfer(engine, pot, localEv, varargin)
%
% Input:
% pot{e}(ki,kj) = potential on edge between nodes i,j  [we assume e=1 only]
% localEv{i}(k) = Pr(observation at node i | Xi=k)
%
%
% Output:
% bel{i}(k) = P(Xi=k|evidence)
% belE{e}(:) = P(edge e)
% logZ 

% localEvGrid(r,c,j)
nr = engine.nrows;
nc = engine.ncols;
K = engine.nstates;
i = 1;
localEvGrid = zeros(nr, nc, K);
for c=1:nc
  for r=1:nr
    localEvGrid(r,c,:) = localEv{i};
    i = i + 1;
  end
end

if isempty(pot) % disconnected graph
  kernel = ones(K,K);
else
  kernel = pot{1};
end
[belNoCell, bel2NoCell, negloglik] = lattice2_hmm_inf(kernel, localEvGrid);
logZ = -negloglik;

Nnodes = nr*nc;
bel = cell(1, Nnodes);
i = 1;
for c=1:nc
  for r=1:nr
    tmp =  belNoCell(r,c,:);
    bel{i} = tmp(:);
    i = i + 1;
  end
end

[coords, edge] = edge_num_lattice2(nr, nc);
Nedges = (nr-1)*nc + nr*(nc-1);
belE = cell(1, Nedges);
for e=1:size(bel2NoCell,3)
  belE{e} = bel2NoCell(:,:,e);
end
