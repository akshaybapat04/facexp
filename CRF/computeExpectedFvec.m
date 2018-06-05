function F = computeExpectedFvec(f, b, doSum)
% COMPUTEEXPECTEDFVEC Repeat each element of the feature vector weighted by state probability
% function F = computedExpectedFvec(f, b)
%
% f(d,t) for d=1:D feature vector for node t
% b(q,t) for q=1:Q belief for node t
% F(dq,t) = f(d,t)*b(q,t)
%
% function F = computedExpectedFvec(f, b, 1)
% In this case we return F(dq) = sum_t F(dq,t) without unrolling over t.
%
% Example
% Suppose node has 3 states and has observed vector f=[a b]'.
% Then conditional feature vectors for each state are cond(:,q)
%
% a 0 0
% b 0 0
% 0 a 0
% 0 b 0
% 0 0 a
% 0 0 b
%
% So the expected feature vector is sum_q cond(:,q) bel(q) = cond*bel =
%
% a 0 0     q1     a*q1
% b 0 0     q2     b*q1
% 0 a 0  x  q3  =  a*q2   = kron(bel, f) = F = (f(:)*bel(:)')(:)
% 0 b 0            b*q2
% 0 0 a            a*q3
% 0 0 b            b*q3

if nargin < 3, doSum = 0; end

if 0 % example 
D = 3; T = 4; Q = 2;
f = randn(D,T);
b = normalize(rand(Q,T),1);
end

if doSum
  %F = kronSumC(b,f);
  F = kronSum(b,f); 
  return;
end


[D T] = size(f);
[Q T] = size(b);

% fast, memory intensive method
fT = repmatC(reshape(f, [D 1 T]), [1 Q 1]);
bT = repmatC(reshape(b, [1 Q T]), [D 1 1]);
F = fT .* bT;
F = reshape(F, D*Q, T);

if 0
  % slow method
  Fslow = zeros(D*Q,T);
  for t=1:T
    %Fslow(:,t) = kron(b(:,t), f(:,t));
    outer = f(:,t) * b(:,t)';
    Fslow(:,t) = outer(:);
  end
  assert(approxeq(F, Fslow))
end

if doSum
  % better to use kronSumC, which avoids repmatC
  F = sum(F,2);
end


% Numerical example of kron
if 0
q1=1; q2=2; q3=3; a=4; b=5;
F = [a 0 0; b 0 0; 0 a 0; 0 b 0; 0 0 a; 0 0 b];
bel = [q1 q2 q3]';
f = [a b]';
assert(isequal(F*bel, kron(bel,f)))
outer = f(:)*bel(:)'; 
assert(isequal(F*bel, outer(:)))
end


