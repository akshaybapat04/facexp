function net = crftrainLabelPots(net, t)
% CRFTRAINLABELPOTS Train potential on discrete edges between hidden  labels
% function net = crftrainLabelPots(net, t)
% 
% t(s,i) = target value (in 1..numStates(i)) for node i in case s
%
% X1 - X2 - X3
% phi(X1,X2) * phi(X2,X3) = P(X1,X2)        P(X2,X3)
%                           ----------     -----------
%                           P(X1) P(X2)     P(X2) P(X3)
% phi{e}(i,j) = #(Xi=i,Xj
