function net = crfunpak(net, w)
% crfunpak Store parameter vector into crf
% function net = crfunpak(net, w)

bs = net.nparamsPerNodeEclass; % block size
wi = w(1:sum(bs));
we = w(sum(bs)+1:end);

for ec=1:net.nnodeEclasses
  i = net.representativeNode(ec);
  if net.clampWeightsForOneState
    Q = net.nstates(i)-1;
  else
    Q = net.nstates(i);
  end
  net.w{ec} = reshape(wi(block(ec,bs)), net.inputDims(i), Q);
end

adjustableEdges = find(net.adjustableEdgeEclassBitv);
bs = net.nstatesPerEdgeEclass(adjustableEdges); % block size
for ec=adjustableEdges(:)' 
  ec2 = find_equiv_posns(ec, adjustableEdges);
  rep = net.representativeEdge(ec,:);
  i = rep(1); j = rep(2);
  pot = exp(we(block(ec2,bs)));
  net.pot{ec} = reshape(pot, net.nstates(i), net.nstates(j));
end
