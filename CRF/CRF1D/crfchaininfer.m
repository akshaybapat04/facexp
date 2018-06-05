function [bel, belE, logZ] = crfchainInfer(net, x)
% CRFCHAININFER Infer probability of hidden nodes given observed features
% function [bel, belE, logZ] = crfchaininfer(net, x)
%
% x(:,t)
% bel(q,t)
% bel(qt,qt+1,t) t=1:T-1


logLocalEv = (net.w' * x); % localEv(q,t)
localEv = exp(logLocalEv);
[bel, belE, logZ] = bpchainInfer(net.infEngine, net.pot, localEv);

