function [labels1D,features1Dcell,nr,nc,D]=crf2DdemoGetData(data,nsamples,doDisplay)
% This function loads or generates examples from one of three 'easy' data sets

% First load or generate the appropriate data

if strcmp(data,'digits')==1
    % Digit Segmentation Data
    assert(nsamples <= 1000);
    nr = 16;
    nc = 16;
    dampen = 3;
    obs = zeros(16,16,nsamples);
    labels = zeros(16,16,nsamples);
    load digits.mat
    for i = 1:nsamples
        obs(:,:,i) = reshape(X(i,:),16,16);
        labels(:,:,i) = 1+(obs(:,:,i)>.5);
        obs(:,:,i) = obs(:,:,i)+abs(randn(16,16))/dampen;
    end
    D=1;
elseif strcmp(data,'lines')==1
    % Binary Denoising Data
    assert(nsamples <= 6);
    nc = 32;
    nr = 32;
    dampen = 2;
    obs = zeros(32,32,nsamples);
    labels = zeros(32,32,nsamples);
    for i = 1:6
        I = double(imread(sprintf('%d.png',i)))/255;
        labels(:,:,i) = -I(:,:,1)+2;
        obs(:,:,i) = I(:,:,1)+abs(randn(32,32))/dampen;
    end
    D=1;
else
    % Generate Data from a Real CRF Potential Model
    p = 0.75;
    kernel = [p 1-p; 1-p p];
    nr = 7;
    nc = 7;
    mu = [100 150];
    Sigma = ones(1,1,2);
    Sigma(:,:,1) = 20;
    Sigma(:,:,2) = 20;
    % Change the below lines to generate different models
    %seed = 1;
    %rand('state', seed); randn('state', seed);
    [labels, obs] = lattice2_hmm_sample(kernel, nr, nc, nsamples, mu, Sigma);
    D=1;
end

% Now put it into a cell array

Nnodes = nr*nc;
features1Dcell = cell(nsamples, Nnodes);
labels1D = zeros(nsamples, Nnodes);

for s=1:nsamples
    i = 1;
    for c=1:nc
        for r=1:nr
            features1Dcell{s,i} = obs(r,c,s);
            labels1D(s,i) = labels(r,c,s);
            i = i + 1;
        end
    end
    
    % Optionally display the images
    
    if nargin > 2 && doDisplay
        figure(1); colormap(gray(256)); imagesc(labels(:,:,s)); colorbar
        figure(2); imagesc(obs(:,:,s)); colormap(gray(256)); colorbar
        drawnow
        pause
    end
end