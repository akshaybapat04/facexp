% Compare exact and BP inference on a binary 2D lattice.


clear all
nstates = 2;

nr = 7;
nc = 7;
Nnodes = nr*nc;
nrows = nr;
ncols = nc;
Nedges = (nrows-1)*ncols + nrows*(ncols-1);
G = mk_2D_lattice(nr, nc, 4);
%G = zeros(Nnodes, Nnodes);

p = 0.9;
kernel = [p 1-p; 1-p p];
kernel = rand(2,2);
kernel = (kernel+kernel') % mk symmetric

% mu(d,q) for dimension d state q (q=1,2)
mu = [1.5 1.5
      1.0 2.0
      1.0 2.0];
Ndims = size(mu,1);
Sigma = zeros(Ndims, Ndims, nstates);
Sigma(:,:,1) = diag(1*[0.5 2 1.5]); 
Sigma(:,:,2) = diag(5*[0.5 2 1.5]); 

nsamples = 20;
% labels(r,c,s), obs(r,c,s,:)
[labels, obs] = lattice2_hmm_sample(kernel, nr, nc, nsamples, mu, Sigma);
labels01 = -(labels-2);
% obs(:,r,c,s)
obs = permute(obs, [4 1 2 3]);
data = reshape(obs, Ndims, nrows*ncols*nsamples);

localEv = zeros(nrows, ncols, nstates, nsamples);
for q=1:nstates
  ev = gaussian_prob(data, mu(:,q), Sigma(:,:,q));
  localEv(:,:,q,:) = reshape(ev, [nrows ncols nsamples]);
end

if 0
  % introduce a non-homogeneity in the observation process
s = 1;
%localEv(:,:,:,s) = 1;
r=2;c=3;
p = 0.9;
if labels(r,c,s)==2
  localEv(r,c,:,1) = [p;1-p];
else
  localEv(r,c,:,1) = [1-p;p];
end
end

probExact = []; probBP = [];
for s=1:nsamples
  [belExact, bel2, negloglik] = lattice2_hmm_inf(kernel, localEv(:,:,:,s));
  [belBP, niter, msgs] = bp_mrf2_lattice2(kernel, localEv(:,:,:,s));
  fprintf('s=%d, niter=%d\n', s, niter)
  
  tmp = belExact(:,:,1); probExact = [probExact; tmp(:)];
  tmp = belBP(:,:,1); probBP = [probBP; tmp(:)];
  
  if 1
  figure(1); clf;imagesc(labels01(:,:,s)); colormap(gray(256)); colorbar; title('truth');
  figure(2); clf; imagesc(belExact(:,:,1)); colormap(gray(256)); colorbar; title('exact');
  figure(3); clf; imagesc(belBP(:,:,1)); colormap(gray(256)); colorbar; title('bp');
  end
  
  if 0
  for d=1:Ndims
    figure(2+d)
    imagesc(squeeze(obs(d,:,:,s)));
    colormap(gray(256))
    colorbar
    title(sprintf('obs dim %d', d))
  end
  end
  drawnow
  %pause
end

[faRateExact, dRateExact] = plotROC(probExact, labels01(:));
[faRateBP, dRateBP] = plotROC(probBP, labels01(:));
figure;
plot(faRateExact, dRateExact, 'ro-');
hold on
plot(faRateBP, dRateBP, 'bx-');
legend('exact','BP')
