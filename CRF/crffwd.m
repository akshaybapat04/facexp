function bel = crffwd(net, x)
% crffwd Compute probability of output nodes given inputs
% function bel = crffwd(net, x)
%
% x{s,i}(:)  inputDims(i) x 1 column vector
% bel{s,i}(:) nstates(i) x 1 column vector 

[ncases nnodes] = size(x);
bel = cell(ncases, nnodes);
for s=1:ncases
  bel(s,:) = crfinfer(net, x(s,:));
end
