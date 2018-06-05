function [g, gdata, gprior] = crfgrad(net, x, t)
% crfgrad CRF gradient function = gradient of negative complete-data log-likelihood
% function [g, gdata, gprior, gs] = crfgrad(net, x, t)
%
% x{s,i}(:) - cell for node i case s is a column feature vector
% t(s,i) - each ROW is the set of desired discrete output labels (targets)
%
% This function can be used by netlab's optimization  routines
% This requires inference so is slow!

% Inference means computing the marginals and logZ.
% logZ (which is slow to compute) is only needed for crferr, not crfgrad.
% The code is explained in crfErrAndGrad.


[ncases nnodes] = size(t);
ns = net.nstates;
gs = zeros(ncases, net.nparamsAdjustable);
infName = sprintf('%sInfer', net.infEngineName);
for s=1:ncases
  [localEv, logLocalEv] = crfMkLocalEv(net, x(s,:)); 
  pot = crfMkPot(net);
  [bel, belE] = feval(infName, net.infEngine, pot, localEv);

  bsi = net.nparamsPerNodeEclass;
  gI = zeros(sum(bsi),1);
  adjustableEdges = find(net.adjustableEdgeEclassBitv);
  bse = net.nstatesPerEdgeEclass(adjustableEdges);
  gE = zeros(sum(bse),1);

  for i=1:nnodes
    ec = net.eclassNode(i);
    if length(net.w{ec})>0
      trueBel = zeros(ns(i),1);
      trueBel(t(s,i)) = 1;
      if net.clampWeightsForOneState
	trueBel = trueBel(2:end);
	bel{i} = bel{i}(2:end);
      end
      clampedFvec = computeExpectedFvec(x{s,i}, trueBel);
      expectedFvec = computeExpectedFvec(x{s,i}, bel{i});
      gI(block(ec,bsi)) = gI(block(ec,bsi)) + (clampedFvec - expectedFvec);
    end
    for j=i+1:nnodes
      e = net.E(i,j);
      if e>0 
	ec = net.eclassEdge(e);
	ec2 = find_equiv_posns(ec, adjustableEdges);
	if ~isempty(ec2)
	  trueBel = zeros(ns(i), ns(j));
	  trueBel(t(s,i), t(s,j)) = 1;
	  clampedFvec = trueBel(:);
	  expectedFvec = belE{e}(:);
	  gE(block(ec2,bse)) = gE(block(ec2,bse)) + (clampedFvec - expectedFvec);
	end
      end
    end
  end
  gs(s,:) = -[gI; gE]'; % minimize NEG log lik
  %fprintf('crfGrad, s=%d\n', s); gs(s,:)
  assert(~any(isnan(gs(s,:))))
end
gdata = sum(gs,1);
[g, gdata, gprior] = gbayes(net, gdata);
drawnow % allow for keyboard interrupt

if 1
%fprintf('crfgrad\n');
gdata;
net.pot{1};
net.w{1};
end
