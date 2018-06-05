function [g] = crfJacobian(w, net, x, t)
% CRFERRANDGRAD CRF error (negative complete-data log-likelihood) and gradient
% function [e, g, es, gs] = crfErrAndgrad(w, net, x, t)
%
% w - parameters of CRF; if [], we assume net contains up-to-date parameters
% x{s,i}(:) - cell for node i case s is a column feature vector
% t(s,i) - each ROW is the set of desired discrete output labels (targets)
% 
% e = scalar, g = row vector
%
% This function can be used by Matlab's optimization  routines
% which require that e and g both be returned together,
% whereas netlab requires e and g to be computed by different functions.
%
% This requires inference so is slow!

% e = sum_s es(s)
% es(s) = - log p(t(s,:) | x(s,:))
%  = - [sum_i <w{i}(:,t(s,i)), x{s,i}(:)> 
%             + sum_{e} log phi{e}(t(s,i), t(s,j)) - log Z(s)]
%  = - [sum_i eI(s,i) + sum_e eE(s,w) - logZ(s) ]
%
% g = sum_s gs(s,:) = row vector
% gs(s,p) = - d/dw(p) log p(t(s,:) | x(s,:)) where p is the p'th parameter
% The gradient for a given term is expecttd-observed feature vector
% For node i, this is  E_q <wi(q),xi> - <wi(obs),xi> +
% For edge e, this is E_[qi,qj] logphi(e)(qi,qj)) - logphi{e}(qi(obs), qj(obs))
%

computeError = 1;
computeLogZ = computeError; % logZ not needed for gradient, and is slow

if ~isempty(w)
  net = crfunpak(net, w);
end

[ncases nnodes] = size(t);
ns = net.nstates;



infName = sprintf('%sInfer', net.infEngineName);
for s=1:ncases
  [localEv, logLocalEv] = crfMkLocalEv(net, x(s,:)); 
  pot = crfMkPot(net);
  if computeLogZ
    [bel, belE, logZ(s)] = feval(infName, net.infEngine, pot, localEv);
  else
    [bel, belE] = feval(infName, net.infEngine, pot, localEv);
  end
  bsi = net.nparamsPerNodeEclass; % block size nodes
  gI = zeros(sum(bsi),1);
  bse = net.nstatesPerEdgeEclass; % block size edges
  gE = zeros(sum(bse),1);
  eE = zeros(1, net.nedges);
  eI = zeros(1, net.nnodes);
  for i=1:nnodes
    ec = net.eclassNode(i);
    trueBel = zeros(ns(i),1);
    trueBel(t(s,i)) = 1;
    if net.clampWeightsForOneState
      trueBel = trueBel(2:end);
      bel{i} = bel{i}(2:end);
    end
    clampedFvec = computeExpectedFvec(x{s,i}, trueBel);
    expectedFvec = computeExpectedFvec(x{s,i}, bel{i});
    gI(block(ec,bsi)) = gI(block(ec,bsi)) + (clampedFvec - expectedFvec);
    %gI{i} = clampedFvec - expectedFvec;
    weights = net.w{ec};
    eI(i) = sum(weights(:) .* clampedFvec(:)); 
    assert(approxeq(logLocalEv{i}(t(s,i)), eI(i)))
    for j=i+1:nnodes
      e = net.E(i,j);
      if e>0
	ec = net.eclassEdge(e);
	trueBel = zeros(ns(i), ns(j));
	trueBel(t(s,i), t(s,j)) = 1;
	clampedFvec = trueBel(:);
	expectedFvec = belE{e}(:);
	%gE{e} = clampedFvec - expectedFvec; % see below
	gE(block(ec,bse)) = gE(block(ec,bse)) + (clampedFvec - expectedFvec);
	%weights = log(net.pot{e}+(net.pot{e}==0)*eps);
	pot = net.pot{ec};
	weights = log(pot);
	eE(e) = sum(weights(:) .* clampedFvec(:)); 
	assert(approxeq(weights(t(s,i), t(s,j)), eE(e)))
      end
    end
  end
  % extract gradient only for adjustable params
  adjustableEdges = find(net.adjustableEdgeEclassBitv);
  adjBlocks = block(adjustableEdges, bse);
  gE = gE(adjBlocks);
  
  gs(s,:) = -[gI; gE]'; % minimize NEG log lik
  %fprintf('crfErrAndGrad, s=%d\n', s); gs(s,:)
  assert(~any(isnan(gs(s,:))))
   if computeError
     es(s) = sum([eI(:)' eE(:)']) - logZ(s);
     es(s) = -es(s);
   end
end
g = sum(gs,1);
e=sum(es)
if computeError
  e = sum(es);
else
  e = [];
end


%%%%%%%%

% If feature vectors for hidden nodes are delta functions,
% the expected feature vector for an edge is just the joint probability.
% Example
% Consider 2 discrete states qi,qj. Each feature vector is stored in F(:,qiqj).
% Then expected vector is F(:,qiqj) * bel(qi,qj) =  bel(qi,qj)
%
% 1 0 0 0     q11   q11
% 0 1 0 0  x  q21 = q21
% 0 0 1 0     q12   q12
% 0 0 0 1     q22   q22
%
% Suppose the observed values are (2,1).
% Then the gradient becomes
%
% 0     q11  
% 1  -  q21 
% 0     q12 
% 0     q22 
