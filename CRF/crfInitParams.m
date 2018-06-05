function net = crfInitParams(net)

nstates = net.nstates;
for ec=1:net.nnodeEclasses
  i = net.representativeNode(ec);
  if net.clampWeightsForOneState
    Q = net.nstates(i)-1;
  else
    Q = net.nstates(i);
  end
  net.w{ec} = 0.1*randn(net.inputDims(i), Q);
  net.nparamsPerNodeEclass(ec) = net.inputDims(i) * Q;
end

net.nstatesPerEdgeEclass = zeros(1, net.nedgeEclasses);
for ec=1:net.nedgeEclasses
  rep = net.representativeEdge(ec,:);
  i = rep(1); j = rep(2);
  %net.pot{e} = normalize(ones(nstates(i), nstates(j))); % should break symmetry
  pot = 0.1*rand(nstates(i), nstates(j));
  net.pot{ec} = pot+pot'; % mk symmetric
  net.nstatesPerEdgeEclass(ec) = nstates(i)*nstates(j);
end

