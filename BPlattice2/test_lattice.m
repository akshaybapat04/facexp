% Make sure lattice code gives same result as general code for sum-product

nrows = 5;
ncols = 5;
npatches = 3;
I = mk_mondrian(nrows, ncols, npatches) + 1;

nstates = npatches+1;
p = 0.7; % prob have same label as your neighbor
%kernel = softeye(nstates, p); 
kernel = rand(nstates,nstates); kernel = kernel + kernel';

q = 0.8; % prob observe correct label
obs_model = softeye(nstates, q);
noisyI = multinomial_sample(I, obs_model);

npixels = nrows*ncols;
local_evidence = multinomial_prob(noisyI, obs_model);
local_evidence2 = myreshape(local_evidence', [nrows, ncols, nstates]);



% Use vectorized code specialized for 2D lattices 
tic; [bel, niter, msgs] = bp_mrf2_lattice2(kernel, local_evidence2, 'method', 'vectorized'); toc

% Use serial code specialized for 2D lattices 
tic; [bel2, niter2] = bp_mrf2_forloops(kernel, local_evidence2); toc
%assert(approxeq(bel_true(:)',bel2(:), 1e-3))
assert(approxeq(bel(:)',bel2(:), 1e-3))


if 1
% Use code for any graph structure  - works because kernel is symmetrice
%tic; [bel3, niter3, msgs3, edge_id] = bp_mrf2_general(adj_mat, kernel, local_evidence); toc
%bel3 = reshape(bel3', [nrows ncols nstates]);
%assert(approxeq(bel(:)',bel3(:), 1e-3))

local_evidence_cell = num2cell(local_evidence, 1);
adj_mat = mk_2D_lattice(nrows, ncols);

tic; [bel3, niter3, msgs3, edge_id] = bp_mrf2_general(adj_mat, kernel, local_evidence_cell); toc
i = 1;
for c=1:ncols
  for r=1:nrows
    b = bel(r,c,:);
    assert(approxeq(bel3{i}, b(:)))
    i = i + 1;
  end
end
end




if 1
  % Use 1 strip is the same as fully vectorized
tic; [bel4, niter4] = bp_mrf2_strips(kernel, local_evidence2, 'nstrips', 1); toc
assert(approxeq(bel(:)',bel4(:), 1e-10))

% Using several strips gives same results
tic; [bel5, niter5] = bp_mrf2_strips(kernel, local_evidence2, 'nstrips', ncols); toc
assert(approxeq(bel(:)',bel5(:), 1e-10))
end

if 0
%tic; [bel6, niter6] = bp_mrf2_lattice_local(kernel, local_evidence2, 'nstrips', 2, 'max_iter',niter); toc
tic; [bel6, niter6] = bp_mrf2_local(kernel, local_evidence2, 'nstrips', 2); toc
assert(approxeq(bel(:)',bel6(:), 1e-10))


% Use C code specialized for 2D lattices 
% Might not give exactly the same results due to rounding errors
tic; [bel6, niter6] = bp_mrf2_C(kernel, local_evidence2); toc
assert(approxeq(bel(:)',bel6(:), 1e-3))
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check pairwise

belp = pairwise_bel_lattice2(kernel, bel, msgs);

%bel_gen = reshape(bel3, [nrows*ncols nstates])';
belp3 = pairwise_bel_general(adj_mat, kernel, bel3, msgs3, edge_id);

[coords, edge] = edge_num_lattice2(nrows, ncols);
for e=1:Nedges
  r1 = coords(e,1); c1 = coords(e,2);  r2 = coords(e,3); c2 = coords(e,4);
  i = sub2ind([nrows ncols], r1, c1);
  j = sub2ind([nrows ncols], r2, c2);
  assert(approxeq(belp3{i,j}, belp(:,:,e)))
end

F = bethe_mrf2_lattice(bel, belp, kernel, local_evidence2);
F3 = bethe_mrf2_general(adj_mat, bel3, belp3, kernel, local_evidence_cell);
assert(approxeq(F, F3))
