function [e, edata, eprior] = crfchainerr(net, features, labels)
% crfchainerr CRF error function = negative complete-data log-likelihood
% function [e, edata, eprior, es] = crfchainerr(net, features, labels)
%
% This function can be used by netlab's optimization  routines.
% This requires inference so is slow!


ncases = length(features);
es = zeros(1,ncases);
for s=1:ncases
  x = features{s};
  lab = labels{s};
  if any(lab==0)
    error('target labels must be in range 1,2,...; zero not allowed')
  end
  logLocalEv = (net.w' * x); % localEv(q,t)
  localEv = exp(logLocalEv);
  [bel, belE, logZ] = bpchainInfer(net.infEngine, net.pot, localEv);
  
  [Q T] = size(bel);
  % eN = sum_t log localEv(labels(t), t)
  ndx = sub2ind([Q T], lab, 1:T);
  eN = sum(logLocalEv(ndx));
  % eE = sum_t log pot(labels(t), labels(t+1))
  ndx2 = sub2ind([Q Q], lab(1:T-1), lab(2:T));
  logpot = log(net.pot);
  eE = sum(logpot(ndx2));

  es(s) = (eN+eE) -logZ;
  es(s) = -es(s); % neg log lik
  assert(~any(isnan(es)))
end
edata = sum(es);

[e, edata, eprior] = errbayes(net, edata);

assert(~isnan(e))
drawnow % allow for keyboard interrupt

