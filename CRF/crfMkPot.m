function pot = crfMkPot(net)
% crfMkPot Copy tied potentials so there is one potential per edge
% function pot = crfMkPot(net)

pot = cell(1, net.nedges);
for i=1:net.nnodes
  for j=i+1:net.nnodes
    e = net.E(i,j);
    if e>0
      ec = net.eclassEdge(e);
      pot{e} = reshape(net.pot{ec}, net.nstates(i), net.nstates(j));
    end
  end
end
