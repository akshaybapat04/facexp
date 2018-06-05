function [e, g] = crfchainErrAndGrad(w, net, features, labels)
% CRFCHAINERRANDGRAD CRF error (negative complete-data log-likelihood) and gradient
% function [e, g] = crfErrAndgrad(w, net, features, labels)
%
% w - parameters of CRF; if [], we assume net contains up-to-date parameters
% features{s}(:,t) - cell for time t case s is a column feature vector
% labels{s}(t) - desired discrete labels (targets) for time t, case s
% 
% e = scalar, g = row vector
%
% This function can be used by Matlab's optimization  routines
% which require that e and g both be returned together.
%
% This requires inference so is slow!

if ~isempty(w)
  net = crfchainunpak(net, w);
end

ncases = length(features);
gs = zeros(ncases, net.nparams);
es = zeros(ncases,1);
for s=1:ncases
  x = features{s};
  [D T] = size(x);
  lab = labels{s};
  logLocalEv = (net.w' * x); % localEv(q,t)
  localEv = exp(logLocalEv);
  [bel, belE, logZ] = bpchainInfer(net.infEngine, net.pot, localEv);
  [Q T] = size(bel);

  ndx = sub2ind([Q T], lab, 1:T);
  trueBel = zeros(Q,T);
  trueBel(ndx) = 1;
  ndx2 = sub2ind([Q Q T-1], lab(1:T-1), lab(2:T), 1:(T-1));
  trueBelE = zeros(Q,Q,T-1);
  trueBelE(ndx2) = 1;

  doSum = 1;
  expectedFvec = computeExpectedFvec(x, bel, doSum);
  clampedFvec = computeExpectedFvec(x, trueBel, doSum);
  gN = clampedFvec - expectedFvec;

  gE = sum(trueBelE - belE,3);
  gE = gE(:);
  gs(s,:) = -[gN; gE]'; % minimize NEG log lik
  assert(~any(isnan(gs(s,:))))
  
  % eN = sum_t log localEv(labels(t), t)
  eN = sum(logLocalEv(ndx));
  ndx2 = sub2ind([Q Q], lab(1:T-1), lab(2:T));
  logpot = log(net.pot);
  eE = sum(logpot(ndx2));
  es(s) = (eN+eE) -logZ;
  es(s) = -es(s);
  assert(~any(isnan(es)))
end
gdata = sum(gs,1);
[g, gdata, gprior] = gbayes(net, gdata);

edata = sum(es);
[e, edata, eprior] = errbayes(net, edata);

drawnow % allow for keyboard interrupt

