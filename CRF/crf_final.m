% Generate data from an HMM with Gaussian outputs.
% By making the centers closer, the classification problem becomes harder.
% Compare 4 classifiers
% -isolated, generative (mixtures of Gaussians)
% -temporal, generative (HMM with Gaussian (not mixGauss) output)
% -isolated, discriminatve (logistic regression)
% -temporal, discriminative (1D CRF)


clear;
rng('default');
nstates = 2; % 8-ary classification
initStateGen = 5*ones(2,1);
p = 0.9;
transMatGen = [p 1-p; 1-p p];

T = 100;
% we train on many sequences and test on 1
Ntrain = 2;
doSplit = 0;
% labelsTrain{s}(t)
labelsTrain = cell(Ntrain,1);
labelsTrain01 = cell(Ntrain,1);
for i=1:Ntrain
  labelsTrain{i} = mc_sample(initStateGen, transMatGen, T);
  labelsTrain01{i} = labelsTrain{i}-1;
end
labelsTest = mc_sample(initStateGen, transMatGen, T);
labelsTest01 = labelsTest-1;

D = 5;
muGen = [ones(D,1) 0.5*ones(D,1)];
SigmaGen = repmat(eye(D), [1 1 nstates]);

% dataTrain{s}(:,t)
dataTrain = cell(Ntrain,1);
dataTrain1 = cell(Ntrain,1);
for i=1:Ntrain
  dataTrain{i} = condgauss_sample(muGen, SigmaGen, labelsTrain{i});
  dataTrain1{i} = [dataTrain{i}; ones(1,T)];
end
dataTest = condgauss_sample(muGen, SigmaGen, labelsTest);
dataTest1 = [dataTest; ones(1,T)];

%%%%%%%%%%%%%%%% train/test HMM

[hmm.initState, hmm.transmat, hmm.mu, hmm.Sigma] = ...
    gausshmm_train_observed(dataTrain, labelsTrain, nstates, 'cov_type', 'diag');
hmm.initState = normalize(ones(nstates,1)); % can't estimate prior from 1 sequence!

localEv = mixgauss_prob(dataTest, hmm.mu, hmm.Sigma);
%assert(all(localEv(:)>0))
%assert(~any(isnan(localEv(:))))
%localEv(:,1:10)

% we do sum-product and take the marginals, not Viterbi decoding
[alpha, ~, gamma, loglik] = fwdback(hmm.initState, hmm.transmat, localEv);
probHMM = gamma(2,:); % state 1= absent,  2 = present
[faRateHMM, dRateHMM] = plotROC(probHMM, labelsTest01);

%%%%%%%%%%% isolated conditional Gaussian

%[iso.mu, iso.Sigma] = mixgaussTrainObserved(dataTrain, labelTrain, nstates);
belGauss = normalize(localEv,1);
probGauss = belGauss(2,:);
[faRateGauss, dRateGauss] = plotROC(probGauss, labelsTest01);


%%%%%%%%%%%%%%%% CRF

D1 = D+1;
regularizer = 1.0;
clamp = 0;
maxIter = 50;
Q = nstates;
% define rnd initial params
w1 = randn(D1,Q);
pot1 = rand(Q,Q);

%%%%%%%%%%%%%%%% train/test using CRF chain code


chain = crfchain(D1, nstates, 'clampWeightsForOneState', clamp, 'alpha', regularizer);
% initialize to same point as general CRF
chain.w = w1;
chain.pot = pot1;


% featuresTrainChain{s}(:,t);
featuresTrainChain = dataTrain1;
featuresTestChain = dataTest1;

tic
chain = crfchaintrain(chain, featuresTrainChain, labelsTrain, 'gradAlgo', 'scg', ...
		      'checkGrad', 'on', 'MaxIter', maxIter, 'verbose', 1);
toc

belChain = crfchaininfer(chain, featuresTestChain);
probCRF = belChain(2,:);
[faRateCRF, dRateCRF] = plotROC(probCRF, labelsTest01);




%%%%%%%%%%%%%%%% train/test using general CRF code (unrolled to fixed length)

% useGeneral = 0; % equivalent results but much slower than chain code

% if useGeneral
G = diag(ones(T-1,1),1); 
%G = zeros(T,T);
%for t=1:T-1
%  G(t,t+1)=1;
%end

% Things get tricky with general CRFs
% because they are designed to handle a fixed number of nodes of variable size,
% whereas HMMs/chains are designed for a variable number of nodes of fixed size.
% So we have to do some shenanigans with cell arrays.

% featuresTrain{s,t}(:)
% labelsTrain(s,t) - regular array
% featuresTest{t}(:)
featuresTrainCRF = cell(Ntrain, T);
labelsTrainCRF = zeros(Ntrain, T);
for s=1:Ntrain
featuresTrainCRF(s,:) = num2cell(dataTrain1{s},1); 
labelsTrainCRF(s,:) = labelsTrain{s};
end
featuresTestCRF = num2cell(dataTest1,1); 

rng('default');
net = crf(repmat(D1,1,T), repmat(nstates,1,T), G, 'eclassNode', ones(1,T), ...
	  'eclassEdge', ones(1,T-1), 'clampWeightsForOneState', clamp, ...
	  'alpha', regularizer);
net.w{1} = w1;
net.pot{1} = pot1;

net.infEngineName = 'bptree';
net.infEngine = bptreeEngine(net.E, net.nstates);


tic
net = crftrain(net, featuresTrainCRF, labelsTrainCRF, 'gradAlgo', 'scg', ...
	       'checkGrad', 'on', 'MaxIter', maxIter);
toc
% this should converge to the same point as chain code
%assert(approxeq(chain.w, net.w{1}))
%assert(approxeq(chain.pot, net.pot{1}))

bel = crfinfer(net, featuresTestCRF);
belCRF = cell2num(bel);
%assert(approxeq(belChain, belCRF)) % hence ROC is same

% end % if general







%%%%%%%%%%%%%% fit isolated logistic classifier

% we just concatenate all the data, since sequence information is irrelevant
dataTrainNoCell = cat(2, dataTrain{:});
labelsTrain01NoCell = cat(2, labelsTrain01{:});
beta = logist2Fit(labelsTrain01NoCell, dataTrainNoCell, 1); % always better to append 1 (offset/bias term)
probLogist = logist2Apply(beta, dataTest);
[faRateLogist, dRateLogist] = plotROC(probLogist, labelsTest01);

%%%%%%%%%%%%% compare


figure;
plot(faRateHMM, dRateHMM, 'ro-');
hold on
plot(faRateCRF, dRateCRF, 'bx-');
plot(faRateLogist, dRateLogist, 'gs-');
plot(faRateGauss, dRateGauss, 'c<-');
legend('HMM','CRF','logist','Gauss')
grid on
axis([0 1 0 1])
title(sprintf('train on %d sequences of length %d in %d dim', ...
	      length(dataTrain), length(dataTrain{1}), D))
drawnow