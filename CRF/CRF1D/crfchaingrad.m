function [g, gdata, gprior] = crfchaingrad(net, features, labels)
% crfchaingrad CRF gradient function = gradient of negative complete-data log-likelihood
% function [g, gdata, gprior] = crfchaingrad(net, features, labels)
%
% features{s}(:,t) - cell for time t case s is a column feature vector
% labels{s}(t) - desired discrete labels (targets) for time t, case s
%
%
% This function can be used by netlab's optimization  routines
% This requires inference so is slow!

ncases = length(features);
gs = zeros(ncases, net.nparams);
for s=1:ncases
  x = features{s};
  [D T] = size(x);
  lab = labels{s};
  logLocalEv = (net.w' * x); % localEv(q,t)
  localEv = exp(logLocalEv);
  [bel, belE] = bpchainInfer(net.infEngine, net.pot, localEv);
  [Q T] = size(bel);

  ndx = sub2ind([Q T], lab, 1:T);
  trueBel = zeros(Q,T);
  trueBel(ndx) = 1;
  ndx2 = sub2ind([Q Q T-1], lab(1:T-1), lab(2:T), 1:(T-1));
  trueBelE = zeros(Q,Q,T-1);
  trueBelE(ndx2) = 1;
  
  %expectedFvecT = computeExpectedFvec(x, bel);
  %clampedFvecT = computeExpectedFvec(x, trueBel);
  %gN = sum(clampedFvecT - expectedFvecT, 2);

  doSum = 1;
  expectedFvec = computeExpectedFvec(x, bel, doSum);
  clampedFvec = computeExpectedFvec(x, trueBel, doSum);
  gN = clampedFvec - expectedFvec;
  
  gE = sum(trueBelE - belE, 3);
  gE = gE(:);
  gs(s,:) = -[gN; gE]'; % minimize NEG log lik
  assert(~any(isnan(gs(s,:))))
end
gdata = sum(gs,1);

[g, gdata, gprior] = gbayes(net, gdata);
