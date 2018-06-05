function X = encodeDiscrete(data, nstates)
% encodeDiscrete Convert matrix of discrete values to cell array of distributed 1-of-K vectors
% function X = encodeDiscrete(data, nstates)
%
% data(s,i) is an integer in [1..nstates(i)]
% X{s,i}(:) is a column vector of size nstates(i)

[ncases nnodes] = size(data);
X = cell(ncases, nnodes);
for i=1:nnodes
  U = unaryEncoding(data(:,i), nstates(i)); % U(:,s)
  X(:,i) = num2cell(U,1)';
end

