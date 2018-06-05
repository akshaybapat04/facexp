function demo_tree()

% Demonstrate belProp on a tree
% Based on code originally written  by Yair Weiss

[G, nstates, Psi, local_ev] = mk_tree1;
nnodes = length(G);

[trueBel, trueBel2, trueNLL, trueMAP] = brute_force_inf_mrf2(G, Psi, nstates, local_ev);


% Direct interface
[bel, niter, msg, edge_id] = bp_mrf2_general(G, Psi, local_ev, 'verbose', 1); % sum product
MAP = bp_mpe_mrf2_general(G, Psi, local_ev, 'verbose', 1); % max product

% Check answers
for i=1:nnodes
  assert(approxeq(bel{i}, trueBel{i}));
end
assert(isequal(MAP, trueMAP))

bel2 = pairwise_bel_general(G, Psi, bel, msg, edge_id);
for i=1:nnodes
  for j=i:nnodes
    if G(i,j)
      assert(approxeq(bel2{i,j}, trueBel2{i,j}))
    end
  end
end

if 0
% Use the BNT interface
mrf2 = mk_mrf2(G, Psi);
engine = belprop_mrf2_inf_engine(mrf2, 'verbose', 1);
[engine, ll, niter] = enter_soft_evidence(engine, local_ev);
for i=1:nnodes
  bel{i} = marginal_nodes(engine, i);
end
MAP = find_mpe(engine, local_ev);
end


%%%%%%%%%%%%%

function [G, nstates, Psi, local_ev] = mk_tree1()

% the graph is
%
% 1 - 2 - 3 - 4
%     |
%     5

nnodes = 5;
G=zeros(nnodes);
for i=1:3;
  G(i,i+1)=1;
  G(i+1,i)=1;
end
G(2,5)=1;G(5,2)=1;

nstates = [3 2 2 2 2];
% we'll assume all nodes are binary except the first one.

% Now make random local evidence
seed = 0;
rand('state', seed);
for i=1:nnodes
  local_ev{i} = rand(nstates(i), 1);
end

% and now make random potentials
for i=1:nnodes
  for j=i+1:nnodes
    if (G(i,j))
      P=rand(nstates(i),nstates(j));
      %Psi{i,j}=P'; % Psi{i,j}(kj,ki) 
      %Psi{j,i}=P;
      Psi{i,j}=P; % Psi{i,j}(ki,kj) 
      Psi{j,i}=P';
    end
  end
end
