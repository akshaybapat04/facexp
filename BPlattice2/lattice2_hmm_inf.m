function [bel, bel2, negloglik] = lattice2_hmm_inf(kernel, local_evidence)
% function [bel, bel2, negloglik] = lattice2_hmm_inf(kernel, local_evidence)
% Exact inference by converting 2D MRF to a 1D HMM
% Input:
% kernel(k1,k2) = potential on edge between nodes i,j (must be the same for all pairs)
% local_evidence(r,c,j) = Pr(observation at node (r,c) | Xi=j); 3D matrix
%
% Output:
% bel(r,c,k) = P(X(r,c)=k|evidence)
% bel2(q1,q2,e) for e given by edge_num_lattice2
% negative log likelihood
%
% This is most efficient if nc >= nr (wide grids)

[nr nc K] = size(local_evidence);
nstates = K^nr;

%[initDist, transMat] = lattice2_to_hmm(kernel, nr);
%[localEvHMM] = lattice2Ev_to_hmmEv(local_evidence);
%[alpha, beta, belHMM, loglik, bel2HMM] = fwdback(initDist, transMat, localEvHMM);
%negloglik = -loglik;

[localPot, pairPot] = lattice2_to_chain(kernel, nr);
[localEvHMM] = lattice2Ev_to_hmmEv(local_evidence);
localEvHMMPot = localEvHMM .* repmat(localPot, 1, nc);

%[alpha, beta, gamma, loglik] = fwdback(ones(nstates,1), mk_stochastic(pairPot), localEvHMM);
[alpha, beta, belHMM, loglik, belEHMM] = fwdback_xi(ones(nstates,1), pairPot, localEvHMMPot);
negloglik = -loglik;


if 0
  % same as fwdback
engine = bpchainEngine;
[belHMM2, belEHMM2, logZ] = bpchainInfer(engine, pairPot, localEvHMMPot);
assert(approxeq(belHMM2, belHMM))
assert(approxeq(belEHMM2, belEHMM))
assert(approxeq(logZ, loglik))
end


if 0
  % We need to use a non-stationary transition matrix
  [initDist, transMat] = lattice2_to_hmm(kernel, nr, nc);
  [alpha, beta, belHMM3, loglik3, belEHMM3] = fwdback(initDist, transMat, localEvHMM, 'act', 1:nc-1);
  assert(approxeq(belHMM3, belHMM))
  assert(approxeq(belEHMM3, belEHMM))
  %assert(approxeq(loglik3, loglik)) % false!
end

%%%%%%%%% convert output of chain back to grid

bigdom = 1:nr;
bigsz = K*ones(1,nr);
bel = zeros(nr, nc, K);

[coords, edge] = edge_num_lattice2(nr, nc);
Nedges = (nr-1)*nc + nr*(nc-1);
bel2 = zeros(K, K, Nedges);
for c=1:nc
  belSlice  = reshape(belHMM(:,c), bigsz); % tmp(X1,X2,X3)
  for r=1:nr
    tmp = marg_table(belSlice, bigdom, bigsz, r);
    bel(r,c,:) = tmp;
    % edge to south
    if r < nr
      i = r;
      j = r+1;
      e = edge(r,c, r+1,c);
      tmp = marg_table(belSlice, bigdom, bigsz, [i j]);
      bel2(:,:,e) = tmp;
    end      
  end
end

% edges to right
bigdom = 1:2*nr;
bigsz = K*ones(1,2*nr);
for c=1:nc-1
  belSlice2 = reshape(belEHMM(:,:,c), bigsz);
  for r=1:nr
    i = r;
    j = r+nr;
    e = edge(r,c, r,c+1);
    tmp = marg_table(belSlice2, bigdom, bigsz, [i j]);
    bel2(:,:,e) = tmp;
  end
end
