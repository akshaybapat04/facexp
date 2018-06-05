% compare general and crfchain code 

clear all

D = 2;
T = 4;
nstates = 2;
dataTrain = randn(D,T);
dataTest = randn(D,T);
labelsTrain = (rand(1,T)>0.5)+1;
labelsTest = (rand(1,T)>0.5)+1;



G = diag(ones(T-1,1),1); 
regularizer = 1;
clamp = 0;
maxIter = 2;

seed = 1;
rand('state', seed); randn('state', seed);
net = crf(repmat(D,1,T), repmat(nstates,1,T), G, 'eclassNode', ones(1,T), ...
	  'eclassEdge', ones(1,T-1), 'clampWeightsForOneState', clamp, ...
	  'alpha', regularizer);
w1 = net.w{1};
pot1 = net.pot{1};

net.infEngineName = 'bptree';
net.infEngine = bptreeEngine(net.E, net.nstates);

featuresTrain = num2cell(dataTrain,1);
featuresTest = num2cell(dataTest,1); % featuresTest{t}(:)

% featuresTrainChain{s}(:,t);
featuresTrainChain = {dataTrain};
featuresTestChain = {dataTest}; 


chain = crfchain(D, nstates, 'clampWeightsForOneState', clamp, 'alpha', regularizer);
% initialize to same point as general CRF
chain.w = w1;
chain.pot = pot1;



[echain] = crfchainerr(chain, featuresTrainChain, {labelsTrain});
[eCRF] = crferr(net, featuresTrain, labelsTrain);
%assert(approxeq(echain, eCRF))
fprintf('echain == eCRF? %d\n', approxeq(echain, eCRF))

[gchain] = crfchaingrad(chain, featuresTrainChain, {labelsTrain});
[gCRF] = crfgrad(net, featuresTrain, labelsTrain);
assert(approxeq(gchain, gCRF))


net = crftrain(net, featuresTrain, labelsTrain(:)', 'gradAlgo', 'scg', ...
	       'checkGrad', 'on', 'MaxIter', maxIter);
w2 = net.w{1};
pot2 = net.pot{1};
bel = crfinfer(net, featuresTest);
belCRF = cell2num(bel);


chain = crfchaintrain(chain, featuresTrainChain, {labelsTrain}, 'gradAlgo', 'scg', ...
		      'checkGrad', 'on', 'MaxIter', maxIter);

% this should converge to the same point, since the inference should be equivalent
assert(approxeq(chain.w, w2))
assert(approxeq(chain.pot, pot2))

belChain = crfchaininfer(chain, featuresTestChain{1});
assert(approxeq(belChain, belCRF))


