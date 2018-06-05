function [F] = bethe_mrf2_general(G, bel, bel2, pot, local_ev)
% function F = bethe_mrf2_general(G, bel, bel2, pot, local_ev)
%
% bel{i}
% bel2{i,j}
% pot{i,j}
% local_ev{i}
%
% free energy F = energy - entropy = -ln Z = -loglik
% Bethe free energy approximates the entropy term.
% Exact Energy E= E2 + E1
%  E2 = -sum_{ij} sum_{x_i,x_j) b_ij(xi,xj) ln pot(xi,xj)
%  E1 = -sum_{i} sum_{x_i) b_i(xi) ln pot(xi)
% Approximate Entropy S = H2 + H1
%  H2 = -sum_{ij} sum_{x_i,x_j) b_ij(xi,xj) ln b_ij(xi,xj)
%  H1 = sum_{i} (qi-1) * sum_{x_i) b_i(xi) ln b_i(xi) %negative entropy
% where qi = num nbrs of node n

if iscell(pot)
  tied_pot = 0;
else
  tied_pot = (ndims(pot)==2);
end

E1 = 0; E2 = 0;
H1 = 0; H2 = 0; 
nnodes = length(G);
for i=1:nnodes
  nbrs = find(G(i,:));
  nnbrs = length(nbrs);
  nbrs = intersect(nbrs, i+1:nnodes);
  %fprintf('i=%d, %d nbrss\n', i, nnbrs)
  b = bel{i}(:);
  b = b + (b==0); % replace 0s by 1s
  H1 = H1 + (nnbrs-1)*(sum(b .* log(b))); 
  E1 = E1 - sum(b .* log(local_ev{i}));
  for j=nbrs(:)'
    if tied_pot
      kernel = pot;
    else
      kernel = pot{i,j};
    end
    b = bel2{i,j}(:);
    b = b + (b==0);
    H2 = H2 - sum(b .* log(b));
    E2 = E2 - sum(b .* log(kernel(:)));
  end
end
    
F = (E1+E2) - (H1+H2);
logZ = -F;
