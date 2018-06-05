% Make an image consisting of numbered regions,
% observe noisy version of numbers, and infer MAP image using BP.
% We use an MRF2 which encourages neighboring sites to have the same region number.

nrows = 100;
ncols = 100;
npatches = 4;
I = mk_mondrian(nrows, ncols, npatches) + 1;
niter = 20*nrows;

nstates = npatches+1;
p = 0.7; % prob have same label as your neighbor
kernel = softeye(nstates, p)


q = 0.8; % prob observe correct label
obs_model = softeye(nstates, q)
noisyI = multinomial_sample(I, obs_model);

npixels = nrows*ncols;
local_evidence = multinomial_prob(noisyI, obs_model);

% Use code specialized for 2D lattices
local_evidence2 = reshape(local_evidence', nrows, ncols, npatches+1);
tic; [MAP, niter] = bp_mpe_mrf2_lattice2(kernel, local_evidence2, ...
					'max_iter',niter, 'method', 'vectorized'); toc

subplot(2,2,1); imagesc(I);  title('true');
subplot(2,2,2); imagesc(noisyI);  title('observed')
subplot(2,2,3); imagesc(MAP); title('BP')
subplot(2,2,4); imagesc(medfilt2(noisyI)); title('medfilt2')

%figure;imagesc(noisyI)
%figure;imagesc(MAP)

tic; [MAP2, niter2] = bp_mpe_mrf2_lattice2(kernel, local_evidence2, ...
					  'max_iter',niter, 'method','strips',  'nstrips', 10); toc
assert(isequal(MAP,MAP2))

tic; [MAP3, niter3] = bp_mpe_mrf2_lattice2(kernel, local_evidence2, ...
					  'max_iter',niter, 'method','local',  'nstrips', 10); toc
%assert(isequal(MAP,MAP3))

tic; [MAP4, niter4] = bp_mpe_mrf2_lattice2(kernel, local_evidence2,  'max_iter',niter, 'method','C'); toc
%assert(isequal(MAP,MAP4)) % tie breaking induces small differences
%assert(length(find(MAP(:) ~= MAP4(:))) < 5) % small Hamming distance
