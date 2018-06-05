function belE = computeEdgeBel(E, pot, bel, msg)

% Compute belief on edges given belief on marginal nodes and messages

nnodes = length(E);
belE = {}; % cell(nnodes, nnodes);
for i=1:nnodes
  nstates(i) = length(bel{i});
end
for i=1:nnodes
  %nbrs = myintersect(find(E(i,:)), i+1:nnodes);
  nbrs = find(E(i,i+1:end))+i; % since starting from i+1
  for j=nbrs(:)'
    %if i<j, e=E(i,j); else e=E(j,i); end
    %kernel = pot{e};
    if i<j
      e = E(i,j);
      kernel = pot{E(i,j)};
    else
      e = E(j,i);
      kernel = pot{E(j,i)}';
    end
    beli = bel{i} ./ msg{E(j,i)};
    belj = bel{j} ./ msg{E(i,j)};
    belij = repmatC(beli(:), 1, nstates(j)) .* repmatC(belj(:)', nstates(i), 1);
    belE{e} = normalize(belij .* kernel);
  end
end
