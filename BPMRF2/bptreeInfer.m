function [bel, belE, logZ] = bptreeInfer(engine, pot, localEv, varargin)
% BPMRF2tree Belief propagation on a tree structured MRF with pairwise potentials
%
% function [bel, belE, logZ] = bpmrf2TreeInfer(engine, pot, localEv, ...)
% Same as bploopyInfer, except we use centralized protocol: leaves to root and back.
%
% Optional arguments (string/value pairs):
% clampedBel - if clampedBel{i} is not empty, we use this distribution instead of inferring it.
%   This can be used to insert (soft) evidence into 'hidden' nodes.

if nargin < 3, localEv = []; end

[clampedBel, doNormalization, computeBethe] = process_options(...
    varargin, 'clampedBel', [], 'doNormalization', 1, 'computeBethe', 0);

E = engine.E;
nnodes = length(E);
Nedges = length(find(E(:)))/2;
msg = cell(1,Nedges); % bi-directional
verbose =  engine.verbose;
nstates = engine.nstates;

bel = cell(1, nnodes);
belE = cell(1,Nedges);


% collect to root
for t=1:nnodes
  i = engine.postorder(t);
  ch = find(engine.tree(i,:)); %children(engine.tree,i);
  pa = find(engine.tree(:,i)); %parents(engine.tree,i);
  %assert(length(pa)<=1)

  if ~isempty(clampedBel) & ~isempty(clampedBel{i})
    bel{i} = clampedBel{i};
  else
    % Compute product of all incoming msgs from children (if any)
    if ~isempty(localEv) & ~isempty(localEv{i})
      bel{i} = localEv{i};
    else
      bel{i} = normalise(ones(nstates(i), 1));
    end
    %bel{i} = normalise(localEv{i});
    for k=ch(:)'
      if verbose, fprintf('%d absorbs from child %d\n', i, k); end
      bel{i} = bel{i} .* msg{E(k,i)};
    end
    if doNormalization
      [bel{i}, littleZ(i)] = normalise(bel{i});  % prevent underflow
    end
  end
   % Pass up to parent (if any)
  if ~isempty(pa)
    j = pa;
    if verbose, fprintf('%d sends to paretn %d\n', i, j); end
    if i<j
      pot_ij = pot{E(i,j)};
    else
      pot_ij = pot{E(j,i)}';
    end
    
    if engine.maximize
      newm = max_mult(pot_ij', bel{i});
    else
      newm = pot_ij' * bel{i};
    end
    msg{E(i,j)} = newm;
  end
end % for i



if computeBethe
  assert(doNormalization)
  % for computing logZ - see betheMRF2 for details (this is exact for trees)
  E1 = 0; E2 = 0;
  H1 = 0; H2 = 0; 
end

% distribute from root 
for t=1:nnodes
  i = engine.preorder(t);
  ch = find(engine.tree(i,:)); % children(engine.tree, i);
  pa = find(engine.tree(:,i)); %parents(engine.tree, i);

  if ~isempty(clampedBel) & ~isempty(clampedBel{i})
    bel{i} = clampedBel{i};
  else
    % Multiply in msg from parent (if any)
    if ~isempty(pa)
      if verbose, fprintf('distrib: %d absorbs from parent %d\n', i, pa); end
      bel{i} = bel{i} .* msg{E(pa,i)};
    end
    if doNormalization
      [bel{i}, normConst(i)] = normalise(bel{i});
    end
  end
  
  if computeBethe
    nnbrs = length(pa) + length(ch);
    b = bel{i}(:);
    b = b + (b==0); % replace 0s by 1s
    H1 = H1 + (nnbrs-1)*(sum(b .* log(b))); 
    if ~isempty(localEv) & ~isempty(localEv{i})
      E1 = E1 - sum(b .* log(localEv{i}));
    end
  end
  
  % Pass down to children (if any)
  if ~isempty(ch)
    for j=ch(:)'
      if verbose, fprintf('distrib: %d sends to child %d\n', i, j); end
      if i<j
	e = E(i,j);
	pot_ij = pot{E(i,j)};
      else
	e = E(j,i);
	pot_ij = pot{E(j,i)}';
      end
      m = msg{E(j,i)}; % msg from child
      m = m + (m==0);
      tmp = bel{i} ./ m;
      if engine.maximize
	newm = max_mult(pot_ij', tmp);
      else
	newm = pot_ij' * tmp; %sum_i pot(i,j) tmp(i) = newmsg(j)
      end
      %newm = normalise(newm);
      msg{E(i,j)} = newm;

      % edge bel
      beli = tmp;
      belj = bel{j};
      belij = repmatC(beli(:), 1, nstates(j)) .* repmatC(belj(:)', nstates(i), 1);
      belE{e} = normalise(belij .* pot_ij);
      
      if computeBethe
	b = belE{e}(:);
	b = b + (b==0);
	H2 = H2 - sum(b .* log(b));
	E2 = E2 - sum(b .* log(pot_ij(:)));
      end
      
      if i > j
	belE{e} = belE{e}'; % make it belE(j,i) - lower numbered node always comes first
      end
    end
  end
end % for i


if length(E)==1
  belE = {};
end

if ~doNormalization
  for i=1:nnodes
    [bel{i}, littleZ] = normalize(bel{i});
  end
  %fprintf('log little Z=%10.9f\n',log(littleZ))
  logZ = log(littleZ);
else
  % as in an HMM, logZ = sum(log(local scaling factors)) 
  % where scaling factors are computed in the upwards pass to the roor
  logZ = sum(log(littleZ));
  if computeBethe
    F = (E1+E2) - (H1+H2);
    assert(approxeq(-F, logZ))
  end
end
