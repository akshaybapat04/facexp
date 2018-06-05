function net = crfchainunpak(net, w)
% crfunpak Store parameter vector into crf
% function net = crfunpak(net, w)

D = net.inputDims;
Q = net.nstates;
wn = w(1:(D*Q));
we = w((D*Q)+1:end);
net.w = reshape(wn, D, Q);
net.pot = reshape(exp(we), Q, Q);
