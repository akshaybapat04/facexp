p = 0.8;
%p=1;
%p=0.5;
kernel = [p 1-p; 1-p p];
mu = [120 130];
Sigma = ones(1,1,2);
Sigma(:,:,1) = 10;
Sigma(:,:,2) = 10;

nr = 8;
nc = 8;
nsamples = 10;
[labels, obs] = lattice2_hmm_sample(kernel, nr, nc, nsamples, mu, Sigma);

for s=1:nsamples
  figure(1); imagesc(labels(:,:,s)); colorbar
  figure(2); image(obs(:,:,s)); colormap(gray(256))
  pause
end
