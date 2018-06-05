% check that BP gives the exact answers on a tree

% the graph is
%
% 1 - 2 - 3 - 4
%     |
%     5

clear all

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
  localEv{i} = rand(nstates(i), 1);
end

[E, Nedges] = assignEdgeNums(G);
pot = cell(1,Nedges);

% and now make random potentials
for i=1:nnodes
  for j=i+1:nnodes
    if (G(i,j))
      P=rand(nstates(i),nstates(j));
      pot{E(i,j)} = P;
    end
  end
end


% Inference


engine = bruteForceMrf2Engine(E, nstates);
[bel, belE, logZ] = bruteForceMrf2Infer(engine, pot, localEv);

engine = bploopyEngine(E, nstates);
[bel2, bel2E, logZ2] = bploopyInfer(engine, pot, localEv);

engine = bptreeEngine(E, nstates, 'verbose', 0);
[bel3, bel3E, logZ3] = bptreeInfer(engine, pot, localEv, 'computeBethe', 1);

engine = bptreeEngine(E, nstates, 'verbose', 0);
[bel4, bel4E, logZ4] = bptreeInfer(engine, pot, localEv, 'doNormalization', 0);

% check invariance to rescaling
pot2 = pot;
for i=1:length(pot)
  pot2{i}= 3*pot{i};
end

engine = bptreeEngine(E, nstates, 'verbose', 0);
[bel5, bel5E, logZ5] = bptreeInfer(engine, pot2, localEv);

engine = bptreeEngine(E, nstates, 'verbose', 0);
[bel6, bel6E, logZ6] = bptreeInfer(engine, pot2, localEv, 'doNormalization', 0);


% Check answers

assert(approxeq(logZ, logZ2))
assert(approxeq(logZ, logZ3))
assert(approxeq(logZ, logZ4))

%assert(approxeq(logZ, logZ5)) % logZ chnages if potentials rescaled
assert(approxeq(logZ5, logZ6))

for i=1:nnodes
  assert(approxeq(bel{i}, bel2{i}))
  assert(approxeq(bel{i}, bel3{i}))
  assert(approxeq(bel{i}, bel4{i}))
  assert(approxeq(bel{i}, bel5{i}))
  assert(approxeq(bel{i}, bel6{i}))
end

for i=1:Nedges
  assert(approxeq(belE{i}, bel2E{i}))
  assert(approxeq(belE{i}, bel3E{i}))
  assert(approxeq(belE{i}, bel4E{i}))
  assert(approxeq(belE{i}, bel5E{i}))
  assert(approxeq(belE{i}, bel6E{i}))
end
