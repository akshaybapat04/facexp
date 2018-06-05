function [F] = bethe_mrf2_lattice(bel, bel2, pot, local_ev)
% function [F] = bethe_mrf2_lattice(bel, bel2, pot, local_ev)
% bel(r,c,:)
% bel2(:,:,r,c,dir) in order east, south, west, north
% pot(xj,xi)
% local_ev(r,c,:)

% See bethe_mrf2_general for details on how to compute Bethe.

[nrows ncols nstates] = size(bel);
ndir = 4; % east, south, west, north
Bel = permute(bel, [3 1 2]); % Bel(states, r, c)
LocalEv = permute(local_ev, [3 1 2]);

E1 = 0; E2 = 0;
H1 = 0; H2 = 0; 

nnbrs = mk_num_nbrs_lattice(nrows, ncols);
for r=1:nrows
  for c=1:ncols
    b = Bel(:,r,c);
    b = b + (b==0); % replace 0s by 1s, valid since 0*log(0) = 0
    H1 = H1 + (nnbrs(r,c)-1)*(sum(b .* log(b))); 
    L = LocalEv(:,r,c);
    L = L + (L==0)*eps; % replace 0 by eps as approximation
    E1 = E1 - sum(b .* log(L));
  end
end

Nedges = (nrows-1)*ncols + nrows*(ncols-1);
for e=1:Nedges
  b = bel2(:,:,e);
  b = b(:) + (b(:)==0);
  H2 = H2 - sum(b .* log(b));
  pot = pot + (pot==0)*eps;   % replace 0 by eps as approximation
  E2 = E2 - sum(b .* log(pot(:)));
end
    
F = (E1+E2) - (H1+H2);
logZ = -F;

