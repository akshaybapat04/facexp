  
function [out_edge, in_edge, nedges] = assign_edge_numbers(nrows, ncols)

% assign each edge in each direction a unique number
north = 1; east = 2; south = 3; west = 4;
ndir = 4; % num directions to send msgs in
out_edge = zeros(nrows, ncols, ndir);
in_edge = zeros(nrows, ncols, ndir);
e = 1;
% north
for r=2:nrows
  for c=1:ncols
    out_edge(r,c,north) = e;
    in_edge(r-1,c,south) = e;
    e = e+1;
  end
end
% east
for r=1:nrows
  for c=1:ncols-1
    out_edge(r,c,east) = e;
    in_edge(r,c+1,west) = e;
    e = e+1;
  end
end
% south
for r=1:nrows-1
  for c=1:ncols
    out_edge(r,c,south) = e;
    in_edge(r+1,c,north) = e;
    e = e+1;
  end
end
% west
for r=1:nrows
  for c=2:ncols
    out_edge(r,c,west) = e;
    in_edge(r,c-1,east) = e;
    e = e+1;
  end
end
nedges = e-1;
