function y=max_mult(A,x)
% MAX_MULT Like matrix multiplication, but sum gets replaced by max
% function y=max_mult(A,x) y(j) = max_i A(j,i) x(i)

%X=ones(size(A,1),1) * x(:)'; % X(j,i) = x(i)
%y=max(A.*X, [], 2);

% This is faster
X=x*ones(1,size(A,1)); % X(i,j) = x(i)
y=max(A'.*X)';
