function [bel, belE, logZ] = bpchainInfer(engine, pot, localEv, varargin)
%
% Inputs:
% engine - we just use engine.maximize
% pot(qt-1, qt) tied edge potential
% localEv(q,t)
%
% Outputs:
% bel(q,t)
% belE(q,q,t) for t=1:T-1

%if nargin < 3, localEv = []; end

[doNormalization, computeBethe] = process_options(...
    varargin, 'doNormalization', 1, 'computeBethe', 0);


[Q T] = size(localEv);
bel = zeros(Q,T);
belE = zeros(Q,Q,T-1);
pot2 = pot'; % pot2(qt,qt-1)
scale = zeros(1,T);
fwdMsg = zeros(Q,T); % fwdMsg(:,t) = msg from t to t+1
% forwards
bel(:,1) = localEv(:,1);
if doNormalization
  [bel(:,1), scale(1)] = normalise(bel(:,1));
end
for t=2:T
  if engine.maximize
    fwdMsg(:,t-1) = max_mult(pot2, bel(:,t-1));
  else
    fwdMsg(:,t-1) = pot2 * bel(:,t-1); % sum_{qt-1} pot(qt, qt-1) bel(qt-1)
  end
  bel(:,t)= localEv(:,t) .* fwdMsg(:,t-1);
  if doNormalization
    [bel(:,t), scale(t)] = normalise(bel(:,t));
  end
end



% backwards
for t=T-1:-1:1
  tmp = bel(:,t+1) ./ fwdMsg(:,t); % undo effect of incoming msg to get beta from gamma
  belttp1 = repmat(bel(:,t), 1, Q) .* repmat(tmp(:)', Q, 1); % alpha * beta
  belE(:,:,t) = normalise(belttp1 .* pot); % pot(qt,qt+1) * belttp1(qt,qt+1)
  if engine.maximize
    backMsg = max_mult(pot, tmp);
  else
    backMsg = pot * tmp; % sum_{qt+1} pot(qt,qt+1) * tmp(qt+1)
  end
  bel(:,t) = bel(:,t) .* backMsg;
  if doNormalization
    bel(:,t) = normalise(bel(:,t));
  end
end


if ~doNormalization
  [junk, littleZ] = normalise(bel(:,1));
  bel = normalise(bel, 1);
  logZ = log(littleZ);
else
  logZ = sum(log(scale));
end

if computeBethe
  % compute logZ - see betheMRF2 for details
  b = bel;
  b = b + (b==0); % replace 0s by 1
  %localEv = localEv + (localEv==0);
  E1 = -sum(sum(b .* log(localEv)));
  H1 = sum(sum(b(:,2:T-1) .* log(b(:,2:T-1)))); % since nnbrs-1 = 0 for both ends
  
  bE = reshape(belE, Q*Q, T-1);
  %bE = bE + (bE==0); % replace 0s by 1
  pot = pot + (pot==0);
  pots = repmat(pot(:), 1, T-1);
  E2 = -sum(sum(bE .* log(pots)));
  H2 = -sum(sum(bE .* log(bE)));
  
  F = (E1+E2) - (H1+H2);
  assert(approxeq(-F, logZ))
end



