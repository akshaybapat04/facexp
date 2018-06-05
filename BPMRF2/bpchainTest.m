% Check inference on a chain gives same results
% using chain or tree code.

Q = 10;
D = 10; % dummy
T = 10; % must be short for tree code to handle efficiently
pot = rand(Q,Q);
localEv = rand(Q,T);
localEvCell = num2cell(localEv,1);

G = diag(ones(T-1,1),1); 
net = crf(repmat(D,1,T), repmat(Q,1,T), G, 'eclassNode', ones(1,T), ...
	  'eclassEdge', ones(1,T-1));
net.pot{1} = pot;
net.w{1} = []; % ignored

potCell = crfMkPot(net);
[bel1cell, belE1cell, logZ1] = bptreeInfer(net.infEngine, potCell, localEvCell);

bel1 = cell2num(bel1cell);
belE1 = zeros(Q,Q,T-1);
for t=1:T-1
  belE1(:,:,t) = belE1cell{t};
end
%belE1 = cell2num(belE1cell);


chain = crfchain(D, Q);
[bel2, belE2, logZ2] = bpchainInfer(chain.infEngine, pot, localEv, ...
				    'doNormalization', 1, 'computeBethe', 1);

[bel3, belE3, logZ3] = bpchainInfer(chain.infEngine, pot, localEv, 'doNormalization', 0);

assert(approxeq(logZ1, logZ2))
assert(approxeq(logZ1, logZ3))

assert(approxeq(bel1, bel2))
assert(approxeq(bel1, bel3))

assert(approxeq(belE1, belE2))
assert(approxeq(belE1, belE3))


