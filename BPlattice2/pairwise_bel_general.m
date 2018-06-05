function bel2 = pairwise_bel_general(adj_mat, pot, bel, msg, edge_id)
% function bel2 = pairwise_bel_general(adj_mat, pot, bel, msg, edge_id)
%
% pot(xi,xj,i,j) or pot{i,j}(ki,kj) or pot(xi,xj) if tied
% If bel{i} is a cell array, bel2{i,j}(xi,xj) is also cell array
% ans so is msgs{e}.
% If bel(:,i) is an  array, bel2(xi,xj,i,j) is also a regular array
% and so is msgs(:,e).


if iscell(bel)
  use_cell = 1;
else
  use_cell = 0;
end

if iscell(pot)
  tied_pot = 0;
else
  tied_pot = (ndims(pot)==2);
end


nnodes = length(adj_mat);
if use_cell
  bel2 = cell(nnodes, nnodes);
  for i=1:nnodes
    nstates(i) = length(bel{i});
  end
  for i=1:nnodes
    nbrs = find(adj_mat(:,i));
    for j=nbrs(:)'
      if tied_pot
	kernel = pot;
      else
	kernel = pot{i,j};
      end
      beli = bel{i} ./ msg{edge_id(j,i)};
      belj = bel{j} ./ msg{edge_id(i,j)};
      belij = repmat(beli(:), 1, nstates(j)) .* repmat(belj(:)', nstates(i), 1);
      bel2{i,j} = normalize(belij .* kernel);
    end
  end
else
  nstates = size(bel, 1);
  bel2 = zeros(nstates, nstates, nnodes, nnodes);
  for i=1:nnodes
    nbrs = find(adj_mat(:,i));
    for j=nbrs(:)'
      if tied_pot
	kernel = pot;
      else
	kernel = pot(:,:,i,j);
      end
      beli = bel(:,i) ./ msg(:,edge_id(j,i));
      belj = bel(:,j) ./ msg(:,edge_id(i,j));
      belij = repmat(beli(:), 1, nstates) .* repmat(belj(:)', nstates, 1);
      bel2(:,:,i,j) = normalize(belij .* kernel);
    end
  end
end
