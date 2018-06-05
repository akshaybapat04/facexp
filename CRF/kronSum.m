function Fout = kronSum(b, f)
% KRONSUM Compute outer product of two vectors and sum over instances
% function Fout = kronSum(b, f)
%
% Fout(dq) = sum_t b(q,t)*f(d,t)


if 0 % test example
D = 10; T = 5000; Q = 10;
f = randn(D,T);
b = normalize(rand(Q,T),1);
tic
F1=kronSum(b,f);
toc
tic
F2=kronSumC(b,f);
toc
assert(approxeq(F1,F2))
end

[D T] = size(f);
[Q T] = size(b);

Fout = zeros(D*Q,T);
for t=1:T
  Fout(:,t) = kron(b(:,t), f(:,t));
end
Fout = sum(Fout,2);

