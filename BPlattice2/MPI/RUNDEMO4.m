% Make an image consisting of numbered regions,
% observe noisy version of numbers, and infer MAP image using BP.
% We use an MRF2 which encourages neighboring sites to have the same region number.

figure;

nrows = 200;
ncols = 400;
npatches = 10;
I = mk_mondrian(nrows, ncols, npatches) + 1;
subplot(4,1,1); imagesc(I);  title('true');

niter = 5*nrows;

nstates = npatches+1;
p = 0.7; % prob have same label as your neighbor
kernel = softeye(nstates, p);


q = 0.8; % prob observe correct label
obs_model = softeye(nstates, q);
noisyI = sample_cond_multinomial(I, obs_model);

subplot(4,1,2); imagesc(noisyI);  title('observed')
pause(2.0);


npixels = nrows*ncols;
local_evidence = eval_pdf_cond_multinomial(noisyI, obs_model);

% Use code specialized for 2D lattices
local_evidence2 = reshape(local_evidence', nrows, ncols, npatches+1);
%tic; [MAP, niter] = bp_mpe_mrf2_lattice(kernel, local_evidence2, ...
%					'max_iter',niter, 'method', 'vectorized'); toc

%tic; [MAP2, niter2] = bp_mpe_mrf2_lattice(kernel, local_evidence2, ...
%					  'max_iter',niter, 'method','strips',  'nstrips', 10); toc

%assert(isequal(MAP,MAP2))

[MAP3, niter3] = bp_mpe_mrf2_lattice(kernel, local_evidence2, ...
					  'max_iter',niter, 'method','local',  'nstrips', 4);
subplot(4,1,3); imagesc(MAP3); title('Serial')
pause(2.0);

%assert(isequal(MAP,MAP3))

[MAP4, niter4] = bp_mpe_mrf2_lattice(kernel, local_evidence2, ...
					  'max_iter',niter, 'method','mpi',  'nstrips', 4);
assert(isequal(MAP3,MAP4))

subplot(4,1,4); imagesc(MAP4); title('Parallel')
%subplot(4,1,4); imagesc(medfilt2(noisyI)); title('medfilt2')

