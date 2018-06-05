function I = mk_mondrian(nrows, ncols, npatches)
% MK_MONDRIAN Make an image containing overlapping, random rectangles
% function I = mk_mondrian(nrows, ncols, npatches)
%
% e.g., mk_mondrian(4, 5, 3)
%  [0 0 1 1 0
%   2 2 1 1 0
%   3 3 3 3 3
%   0 0 1 1 0]

I = zeros(nrows, ncols);
for i=1:npatches
  top_row = sample_discrete(normalise(ones(1,nrows-1)));
  left_col = sample_discrete(normalise(ones(1,ncols-1)));
  num_rows = sample_discrete(normalise(ones(1,nrows-top_row)));
  num_cols = sample_discrete(normalise(ones(1,ncols-left_col)));
  bot_row = top_row + num_rows-1;
  right_col = left_col + num_cols;
  I(top_row:bot_row, left_col:right_col) = i;
end
