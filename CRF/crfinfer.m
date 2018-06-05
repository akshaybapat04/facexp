function [bel, belE, logZ, localEv] = crfinfer(net, x, varargin)
% CRFINFER Infer probability of output nodes given input
% function [bel, belE, logZ] = crfinfer(net, x, ...)
%
% INPUT
% x{i}(:)
% ... optional arguments are passed to net.infEngine
%
% Output:
% bel{i}(q) = P(Yi=q|x)
% belE{e}(qi,qj)
% logZ
% localEv{i}(q)


[Ntrain Nvars] = size(x);
if net.addOneToFeatures
  for s=1:Ntrain
    for i=1:Nvars
      x{s,i} = [x{s,i};1];
    end
  end
end

[localEv] = crfMkLocalEv(net, x); 
infName = sprintf('%sInfer', net.infEngineName);
pot = crfMkPot(net);
[bel, belE, logZ] = feval(infName, net.infEngine, pot, localEv, varargin{:});

