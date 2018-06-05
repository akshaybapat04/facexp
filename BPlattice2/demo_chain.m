function demo_chain()

% Demonstrate belProp on a 1D chain
% We make all nodes have the same state-space and tie params so we can compare
% to the lattice code

[G, nstates, Psi, local_ev] = mk_chain1;
kernel = Psi{1,2}; % tied
nnodes = length(G);

[trueBel, trueBel2, trueNLL, trueMAP] = brute_force_inf_mrf2(G, Psi, nstates, local_ev);

% Check general BP code
[bel, niter, msg, edge_id] = bp_mrf2_general(G, Psi, local_ev, 'verbose', 1); % sum product
MAP = bp_mpe_mrf2_general(G, Psi, local_ev, 'verbose', 1); % max product

for i=1:nnodes
  assert(approxeq(bel{i}, trueBel{i}));
  tbel(i,:) = trueBel{i}';
  if i<nnodes
    tbel2(:,:,i) = trueBel2{i,i+1};
  end
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

NLL = bethe_mrf2_general(G, bel, bel2, Psi, local_ev);
assert(approxeq(NLL, trueNLL))



% Lattice code - 1 row
local_ev2D = zeros(1, nnodes, nstates(1));
for j=1:nnodes
  local_ev2D(1,j,:) = local_ev{j};
end
check(kernel, local_ev2D, tbel, tbel2, trueNLL);


% Lattice code - 1 col
local_ev2D = zeros(nnodes, 1, nstates(1));
for j=1:nnodes
  local_ev2D(j,1,:) = local_ev{j};
end
check(kernel, local_ev2D, tbel, tbel2, trueNLL);

%%%%%%%%%%%%

function check(kernel, local_ev2D, tbel, tbel2, trueNLL)

% Check brute force lattice code
[bel_bf, bel2_bf, NLL_bf] = brute_force_inf_lattice2(kernel, local_ev2D);
assert(approxeq(tbel, bel_bf))
assert(approxeq(tbel2, bel2_bf))
assert(approxeq(trueNLL, NLL_bf))

[bel_bf, bel2_bf, NLL_bf] = brute_force_inf_lattice2_wrapper(kernel, local_ev2D);
assert(approxeq(tbel, bel_bf))
assert(approxeq(tbel2, bel2_bf))
assert(approxeq(trueNLL, NLL_bf))

% Check BP lattice code
[bel_lattice_vec, niter, msgs] = bp_mrf2_vectorized(kernel, local_ev2D);
bel2 = pairwise_bel_lattice2(kernel, bel_lattice_vec, msgs);
NLL = bethe_mrf2_lattice(bel_lattice_vec, bel2, kernel, local_ev2D);

assert(approxeq(tbel, bel_lattice_vec))
assert(approxeq(tbel2, bel2))
assert(approxeq(NLL, trueNLL))


[bel_lattice_slow] = bp_mrf2_lattice2(kernel, local_ev2D, 'method', 'forloops');
[bel_lattice_strips] = bp_mrf2_lattice2(kernel, local_ev2D, 'method', 'strips');
assert(approxeq(tbel, bel_lattice_slow))
assert(approxeq(tbel, bel_lattice_strips))


%%%%%%%%%%%%%%


%%%%%%%%%%%%%

function [G, nstates, Psi, local_ev] = mk_chain1()

% the graph is
%
% 1 - 2 - 3 - 4


nnodes = 4;
G=zeros(nnodes);
for i=1:3;
  G(i,i+1)=1;
  G(i+1,i)=1;
end

nstates = [2 2 2 2];
% we'll assume all nodes are binary

% Now make random local evidence
seed = 0;
rand('state', seed);
for i=1:nnodes
  local_ev{i} = rand(nstates(i), 1);
end

P=rand(2,2);
P = P+P'; % mk symmetric
% and now make random potentials
for i=1:nnodes
  for j=i+1:nnodes
    if (G(i,j))
      Psi{i,j}=P; % Psi{i,j}(ki,kj) 
      Psi{j,i}=P';
    end
  end
end
