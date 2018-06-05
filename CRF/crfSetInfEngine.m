function net = crfSetInfEngine(net, infEngineName, varargin)

net.infEngineName = infEngineName;
net.infEngine = feval(sprintf('%sEngine',infEngineName), net.E, net.nstates);


