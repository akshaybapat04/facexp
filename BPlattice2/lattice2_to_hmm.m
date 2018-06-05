function [initDist, transMat] = lattice2_to_hmm(kernel, nr, nc)

% Given a homogeneous potential pot(k1,k2) applied to all edges of a 2D MRF,
% compute the NON-STATIONARY  parameters of the corresponding 1D chain 
% by constructing the 2-slice HMM of nr rows, and then marginalizing
% eg for nr = 3, nc=3
% 1 - 4 - 7
% |   |   |
% 2 - 5 - 8 
% |   |   |
% 3 - 6 - 9
% 
% initDist = P(X1,X2,X3)
% transMat{1} = P(X4,X5,X6 | X1,X2,X3)
% transMat{2} = P(X7,X8,X9 | X4,X5,X6)
%
% To see that the transition matrices are non-stationary, consider
% the case of no local evidence.
% A regular HMM gives slices 2-4 the same belief (only slice 1 is aysmmetric)
% whereas in the MRF, bel(:,1)=bel(:,4), and bel(:,2)=bel(:,3) (both ends are asymmetric).


K = length(kernel);
nstates = K^nr;
[localPot, pairPot] = lattice2_to_chain(kernel, nr);
%localEvHMM = ones(nstates,2); % 2 slices
localEvHMM = repmat(localPot, 1, nc);
initDist = ones(nstates,1);
[alpha, beta, belHMM, loglik, belEHMM] = fwdback_xi(initDist, pairPot, localEvHMM);

%assert(approxeq(belHMM(:,1), belHMM(:,2)))
initDist = belHMM(:,1);
for t=1:nc-1
  transMat{t} = mk_stochastic(belEHMM(:,:,t));
end


 
