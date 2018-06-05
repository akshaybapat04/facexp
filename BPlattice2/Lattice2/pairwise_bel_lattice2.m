function bel2 = pairwise_bel_lattice2(pot, bel, msgs)
% function bel2 = compute_pairwise_bel_lattice2(pot, bel, msgs)
% bel2(x1,x2, e)

[nrows ncols nstates] = size(bel);
Bel = permute(bel, [3 1 2]); % Bel(states, r, c)

[coords, edge] = edge_num_lattice2(nrows, ncols);
Nedges = size(coords, 1);
bel2 = zeros(nstates, nstates, Nedges);

if 0
for e=1:Nedges
  r1 = coords(e,1); c1 = coords(e,2);  r2 = coords(e,3); c2 = coords(e,4);
  bela = Bel(:,r1,c1) ./ squeeze(msgs(r1,c1,4,:));
  belb = Bel(:,r2,c2) ./ squeeze(msgs(r2,c2,2,:));
  belab = repmat(bela(:), 1, nstates) .* repmat(belb(:)', nstates, 1);
  bel2(:,:,e) = normalize(belab .* pot);
end
end

for r=1:nrows
  for c=1:ncols
    if c<ncols % east
      r2 = r; c2 = c+1; e = edge(r,c,r2,c2);
      bela = Bel(:, r,c) ./ squeeze(msgs(r,c,3,:)); % from west
      belb = Bel(:, r,c+1) ./ squeeze(msgs(r,c+1,1,:)); % from east
      belab = repmat(bela(:), 1, nstates) .* repmat(belb(:)', nstates, 1);
      bel2(:,:,e) = normalize(belab .* pot);
    end
    if r<nrows % south
      r2 = r+1; c2 = c; e = edge(r,c,r2,c2);
      bela = Bel(:,r,c) ./ squeeze(msgs(r,c,2,:)); % from south
      belb = Bel(:,r+1,c) ./ squeeze(msgs(r+1,c,4,:)); % from north
      belab = repmat(bela(:), 1, nstates) .* repmat(belb(:)', nstates, 1);
      bel2(:,:,e) = normalize(belab .* pot);
    end
  end
end
