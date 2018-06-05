function [localPot, pairPot] = lattice2_to_chain(pot, nr)

% Given a homogeneous potential pot(k1,k2) applied to all edges of a 2D MRF,
% compute the parameters of the corresponding 1D chain 
% by constructing the 2-slice "HMM" of nr rows
% eg for nr = 3
% 1 - 4
% |   |
% 2 - 5
% |   |
% 3 - 6
% 
% localPot([X1,X2,X3]) = pot(X1,X2)*pot(X2,X3)
% pairPot([X1,X2,X3], [X4,X5,X6]) = pot(X1,X4)*pot(X2,X5)*pot(X3,X6)

K = length(pot); % num states
bigDom = 1:(2*nr);
bigSz = K*ones(1,2*nr);
pairPot = myones(bigSz);
for r=1:nr
  smalldom = [r r+nr]; 
  smallsz = [K K];
  pairPot = mult_by_table(pairPot, bigDom, bigSz, pot, smalldom, smallsz);
end
nstates = K^nr;
pairPot = reshape(pairPot, nstates, nstates);

bigDom = 1:nr;
bigSz = K*ones(1, nr);
localPot = myones(bigSz);
for r=1:nr-1
  smalldom = [r r+1]; 
  smallsz = [K K];
  localPot = mult_by_table(localPot, bigDom, bigSz, pot, smalldom, smallsz);
end
localPot = localPot(:);
