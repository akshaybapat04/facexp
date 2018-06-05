function [labels, obs] = lattice2_hmm_sample(kernel, nr, nc, nsamples, mu, Sigma)
%
% [labels] = lattice2_hmm_sample(kernel, nr, nc, nsamples)
% Convert MRF of size nr x nc to a (non-stationary) HMM,
% and then draw 2D labels from this.
%
% [labels, obs] = lattice2_hmm_sample(kernel, nr, nc, nsamples, mu, Sigma)
% We also draw a noisy continuous observations given the label:
%  obs(r,c,s,:) ~ N(mu(:,l), Sigma(:,:,l)) where l=labels(r,c,s)

if nargin < 5
  mu = [];
  Sigma = [];
end

K = length(kernel);

if 0
  % sampling from the marginals is not correct
[bel] = lattice2_hmm_inf(kernel, ones(nr, nc, K));
labels = zeros(nr, nc, nsamples);
for r=1:nr
  for c=1:nc
    tmp = sample_discrete(bel(r,c,:), nsamples, 1);
    labels(r,c,:) = tmp(:);
  end
end
end

[initDist, transMat] = lattice2_to_hmm(kernel, nr, nc);
labels = zeros(nr, nc, nsamples);
samples = zeros(nsamples, nc); % samples(s,t) t=1:nc
sz = K*ones(1,nr);
for s=1:nsamples
  t = 1;
  samples(s,t) = sample_discrete(initDist);
  ndx = ind2subv(sz, samples(s,t));
  labels(:,t,s) = ndx(:);
  for t=2:nc
    samples(s,t) = sample_discrete(transMat{t-1}(samples(s,t-1), :));
    ndx = ind2subv(sz, samples(s,t));
    labels(:,t,s) = ndx(:);
  end
end


if ~isempty(mu)
  D = size(mu,1);
  assert(K==size(mu,2));
  obs = zeros(nr*nc*nsamples, D);
  for k=1:K
    ndx = find(labels==k);
    vals = sample_gaussian(mu(:,k), Sigma(:,:,K), length(ndx));
    obs(ndx,:) = vals;
  end
  obs = reshape(obs, [nr nc nsamples D]);
end
