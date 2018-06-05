function [coords, edge] = edge_num_lattice2(nrows, ncols)
% function [coords, edge] = edge_num_lattice2(nrows, ncols)
% coords(e, :) = [r1 c1 r2 c2]
% edge(r1,c1,r2,c2) = e
% Edges are ordered east then south for each site
% 
% Example: for 2x2 grid, edges numbered as
% x 1 x
% 2   3
% x 4 x
%
% Example: for 3x3 grid, edges numbered as
% x 1 x 3 x
% 2   4   5 
% x 6 x 8 x
% 7   9  10
% x 11x 12x
% 
% [coords, edge] = edge_num_lattice2(3,3)
% coords = 
%      1     1     1     2
%      1     1     2     1
%      1     2     1     3
%      1     2     2     2
%      1     3     2     3
%      2     1     2     2
%      2     1     3     1
%      2     2     2     3
%      2     2     3     2
%      2     3     3     3
%      3     1     3     2
%      3     2     3     3


Nedges = (nrows-1)*ncols + nrows*(ncols-1);
coords = zeros(Nedges, 4);
edge = zeros(nrows, ncols, nrows, ncols);
e = 1;
for r=1:nrows
  for c=1:ncols
    if c<ncols
      coords(e, :) = [r c r c+1];
      edge(r,c,r,c+1) = e;
      e = e+1;
    end
    if r<nrows
      coords(e, :) = [r c r+1 c];
      edge(r,c,r+1,c) = e;
      e = e+1;
    end
  end
end
