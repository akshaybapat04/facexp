% Compare logistic regression and CRF with 1 node (Y|X)
%
% P(y=q|x) propto exp(W(:,q)' * [x;1])
% For the binary case, this reduces to logistic regression:
%
% P(y=2|x) = exp(W^T f(:,2)) / [" + exp(W^T f(:,1)]
%          = 1 / [1 + exp(W^T f(:,1)-f(:,2)) ]
%          = 1 / [1 + exp(W^T f(:,1))*exp(-W^T f(:,2)) ]
%          = 1 / 1[ + exp(-w^T F)) ] = netlab glm fn
% 
% where f(:,1) = [F 0] and f(:,2) = [0 F] and F=[x 1]
% So if we set W=[0 w], we can equate the two methods

clear all
D = 2;
input_dim = D;

% from netlab demglm1

% Fix seeds for reproducible results
randn('state', 42);
rand('state', 42);

ndata = 400;
% Generate mixture of two Gaussians in two dimensional space
mix = gmm(2, 2, 'spherical');
mix.priors = [0.4 0.6];              % Cluster priors 
mix.centres = [2.0, 2.0; 0.0, 0.0];  % Cluster centres
mix.covars = [0.5, 1.0];

[data, label] = gmmsamp(mix, ndata);
targets = label - ones(ndata, 1);

fh1 = figure;
plot(data(label==1,1), data(label==1,2), 'bo');
hold on
axis([-4 5 -4 5])
set(gca, 'box', 'on')
plot(data(label==2,1), data(label==2,2), 'rx')
title('Data')


% Train
net = glm(input_dim, 1, 'logistic');
net.w1 = zeros(size(net.w1));
net.b1 = zeros(size(net.b1));
options = foptions;
maxIter = 50;
options(1) = 1; % verbose
options(2) = 1e-1; % paramTol
options(3) = 1e-1; % errTol
options(14) = maxIter;
net = glmtrain(net, options, data, targets);
w = [net.w1; net.b1]

% Test

x = -4.0:1:5.0;
y = -4.0:1:5.0;
[X, Y] = meshgrid(x,y);
X = X(:);
Y = Y(:);
grid = [X Y];
Z = glmfwd(net, grid);
Z = reshape(Z, length(x), length(y));

Z = reshape(Z, length(x), length(y));
v = [0.1 0.5 0.9];
[c, h] = contour(x, y, Z, v);
title('Generalized Linear Model')
set(h, 'linewidth', 3)
clabel(c, h);


%%%%%%%%%%%%%%%%%%%%%%%
% Now make CRF

G = 1;
nstates = 2;
% must turn off regularization to ensure results are equivalent
netc = crf(input_dim+1, nstates, G, 'alpha', 0);

netc.infEngineName = 'bptree';
netc.infEngine = bptreeEngine(netc.E, netc.nstates);

% Initialize parameters to glm values and check inference
netc.w{1} = [zeros(3,1) w(:)];

for n=1:size(grid,1)
  x = grid(n,:);
  fvec = {[x 1]'};
  bel = crfinfer(netc, fvec);
  p2 = bel{1}(2);
  p1 = glmfwd(net, x);
  assert(approxeq(p1,p2))
end

% Check that, after learning, we get the same results.
% Due to numerical problems, the gradient methods end up in slightly
% different points in parameter space.
% Hence we only check for approximate equality

M = [data ones(ndata,1)]; % M(s,:)
X = num2cell(M',1)'; % X{s}(:)

netc = crf(input_dim+1, nstates, G, 'alpha', 0, 'clampWeightsForOneState', 1);
%netc.w{1} = zeros(size(netc.w{1}));
%netc.w{1} = 0.1*randn(size(netc.w{1}));

netc.infEngineName = 'bptree';
netc.infEngine = bptreeEngine(netc.E, netc.nstates);

netc = crftrain(netc, X, label, 'checkGrad', 'on', 'gradAlgo', 'fminunc', 'maxIter', maxIter);

for n=1:size(grid,1)
  x = grid(n,:);
  fvec = {[x 1]'};
  bel = crfinfer(netc, fvec);
  p2 = bel{1}(2);
  p1 = glmfwd(net, x);
  assert(approxeq(p1,p2, 1e-1))
  %assert(approxeq(p1,p2))
end

