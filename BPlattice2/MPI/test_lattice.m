% Make sure lattice code gives same result as general code for sum-product

nrows = 10;
ncols = 10;
npatches = 2;
I = mk_mondrian(nrows, ncols, npatches) + 1;

nstates = npatches+1;
p = 0.7; % prob have same label as your neighbor
kernel = softeye(nstates, p);


q = 0.8; % prob observe correct label
obs_model = softeye(nstates, q);
noisyI = sample_cond_multinomial(I, obs_model);

npixels = nrows*ncols;
local_evidence = eval_pdf_cond_multinomial(noisyI, obs_model);
local_evidence2 = reshape(local_evidence', nrows, ncols, nstates);

niter = 2;

% Use vectorized code specialized for 2D lattices 
%tic; [bel, niter] = bp_mrf2_lattice_vectorized(kernel, local_evidence2, 'max_iter', niter); toc
tic; [bel, niter] = bp_mrf2_lattice_vectorized(kernel, local_evidence2); toc

% Local strips
%tic; [bel6, niter6] = bp_mrf2_lattice_local(kernel, local_evidence2, 'nstrips', 2, 'max_iter',niter); toc
tic; [bel6, niter6] = bp_mrf2_lattice_local(kernel, local_evidence2, 'nstrips', 2); toc
approxeq(bel(:)',bel6(:), 1e-10)

if 1
% Use 1 strip is the same as fully vectorized
tic; [bel4, niter4] = bp_mrf2_lattice_strips(kernel, local_evidence2, 'nstrips', 1); toc
approxeq(bel(:)',bel4(:), 1e-10)

% Using several strips gives same results
tic; [bel5, niter5] = bp_mrf2_lattice_strips(kernel, local_evidence2, 'nstrips', ncols); toc
approxeq(bel(:)',bel5(:), 1e-10)


% Use serial code specialized for 2D lattices 
tic; [bel2, niter2] = bp_mrf2_lattice_forloops1(kernel, local_evidence2); toc
approxeq(bel(:)',bel2(:), 1e-3)
end

% Use code for any graph structure  
adj_mat = mk_2D_lattice(nrows, ncols, 4);
%tic; [bel3, niter3] = bp_mrf2(adj_mat, kernel, local_evidence, 'max_iter', niter); toc
tic; [bel3, niter3] = bp_mrf2(adj_mat, kernel, local_evidence); toc
bel3 = reshape(bel3', [nrows ncols nstates]);
approxeq(bel(:)',bel3(:), 1e-3)


