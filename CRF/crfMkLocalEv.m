function [localEv, logLocalEv] = crfMkLocalEv(net, x)
% crfMkLocalEv Make local evidence vectors
% function localEv = crfMkLocalEv(net, x)
%
% x{i}(:) - ignored if empty (no evidence)
% localEv{i}(q) = exp [ <x{i} , w{i}(:,q) > ] = exp sum_d x{i}(d) w{i}(d,q)


assert(length(x)==net.nnodes);
localEv = cell(1, net.nnodes);
logLocalEv = cell(1, net.nnodes);
for i=1:net.nnodes
  ec = net.eclassNode(i);
  if isempty(net.w{ec}) | isempty(x{i})
    logLocalEv{i} = zeros(net.nstates(i),1);
    localEv{i} = ones(net.nstates(i), 1);
  else
    if net.clampWeightsForOneState
      W = [zeros(net.inputDims(i),1) net.w{ec}];
    else
      W = net.w{ec};
    end
    data = x{i}(:);
    %if net.addOneToFeatures
    %  data = [data; 1];
    %end
    logLocalEv{i} = (W' * data);
    localEv{i} = exp(logLocalEv{i});
    localEv{i} = normalize(localEv{i}); %%%%%%%%%%%%%%%%
    assert(~any(isnan(localEv{i})))
  end
end

