function [e, edata, eprior] = crferr(net, x, t)
% crferr CRF error function = negative complete-data log-likelihood
% function [e, edata, eprior, es] = crferr(net, x, t)
%
% x{s,i}(:) - cell for node i case s is a column feature vector
% t(s,i) - each ROW is the set of desired discrete output labels (targets)
%
% This function can be used by netlab's optimization  routines
% This requires inference so is slow!

% The code is explained in crfErrAndGrad

[ncases nnodes] = size(t);
if nnodes ~= net.nnodes
  error(sprintf('targets should have size ncases x nnodes'))
end
ns = net.nstates;
es = zeros(1,ncases);
infName = sprintf('%sInfer', net.infEngineName);
for s=1:ncases
  [localEv, logLocalEv] = crfMkLocalEv(net, x(s,:)); 
  pot = crfMkPot(net);
  [bel, belE, logZ(s)] = feval(infName, net.infEngine, pot, localEv);

  eE = zeros(1, net.nedges);
  eI = zeros(1, net.nnodes);
  for i=1:nnodes
     eI(i) = logLocalEv{i}(t(s,i));
    for j=i+1:nnodes
      e = net.E(i,j);
      if e>0
	ec = net.eclassEdge(e);
	pot = net.pot{ec};
	%weights = log(net.pot{e}+(net.pot{e}==0)*eps);
	weights = log(pot);
	eE(e) = weights(t(s,i), t(s,j));
      end
    end
  end
  es(s) = sum([eI(:)' eE(:)']) - logZ(s);
  es(s) = -es(s); % neg log lik
  %fprintf('crfErr, s=%d, e=%5.3f\n', s, es(s));
  assert(~any(isnan(es(s))))
end
edata = sum(es);

[e, edata, eprior] = errbayes(net, edata);

assert(~isnan(e))
drawnow % allow for keyboard interrupt

if 1
%fprintf('crferr\n');
edata;
net.pot{1};
net.w{1};
end
