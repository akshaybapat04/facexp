function  [G, pot, localEv] = mkRndMRF(nnodes, power)


G = randn(nnodes, nnodes) > 0.2;
G = G+G';
G = (G>0);
G = setdiag(G,0);
%figure(1); clf; draw_dot(G); %draw_graph(G);

nstates = 2*ones(1,nnodes);

for i=1:nnodes
  localEv{i} = rand(nstates(i), 1);
end

[E, Nedges] = assignEdgeNums(G);
pot = cell(1,Nedges);
%pot = cell(nnodes, nnodes);

% and now make random potentials
for i=1:nnodes
  for j=i+1:nnodes
    if (G(i,j))
      P=rand(nstates(i),nstates(j));
      P = normalize(P .^ power);
      %q = 0.9;
      %P = [q 1-q; 1-q q];
      pot{E(i,j)} = P;
      %pot{i,j} = P;
    end
  end
end
