% Demonstrate belProp on a tree
% Based on code originally written  by Yair Weiss

% the graph is
%
% 1 - 2 - 3 - 4
%     |
%     5

nnodes = 5;
adjMatrix=zeros(nnodes);
for i=1:3;
  adjMatrix(i,i+1)=1;
  adjMatrix(i+1,i)=1;
end
adjMatrix(2,5)=1;adjMatrix(5,2)=1;

% Now make random local potentials
seed = 0;
rand('state', seed);
% we'll assume all nodes are binary except the first one.
Ls{1}=rand(3,1);
for i=2:5
  Ls{i}=rand(2,1);
end

% and now make random potentials
for i=1:nnodes
  for j=i+1:nnodes
    if (adjMatrix(i,j))
      nI=length(Ls{i});
      nJ=length(Ls{j});
      P=rand(nI,nJ);
      Psi{i,j}=P'; % Psi{i,j}(kj,ki) 
      Psi{j,i}=P;
    end
  end
end

% now that we finally have a problem, lets solve it exactly

[trueBel,trueMAP]=enumerative_inf_mrf2(adjMatrix, Psi, Ls);


% Direct interface
[bel, niter] = bp_mrf2(adjMatrix, Psi, Ls, 'verbose', 1); % sum product
MAP = bp_mpe_mrf2(adjMatrix, Psi, Ls, 'verbose', 1); % max product

% Check answers
for i=1:nnodes
  assert(approxeq(bel{i}, trueBel{i}));
end
assert(isequal(MAP, trueMAP))


if 0
% Use the BNT interface
mrf2 = mk_mrf2(adjMatrix, Psi);
engine = belprop_mrf2_inf_engine(mrf2, 'verbose', 1);
[engine, ll, niter] = enter_soft_evidence(engine, Ls);
for i=1:nnodes
  bel{i} = marginal_nodes(engine, i);
end
MAP = find_mpe(engine, Ls);
end
