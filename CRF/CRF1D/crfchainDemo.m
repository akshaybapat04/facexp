% Generate data from an HMM with Gaussian outputs.
% By making the centers closer, the classification problem becomes harder.
% Compare 4 classifiers
% -isolated, generative (mixtures of Gaussians)
% -temporal, generative (HMM with Gaussian (not mixGauss) output)
% -isolated, discriminatve (logistic regression)
% -temporal, discriminative (1D CRF)


clear all
seed = 1;
rand('state', seed); randn('state', seed);

nstates = 2; % binary classification
initStateGen = [0.5 0.5]';
p = 0.9;
transMatGen = [p 1-p; 1-p p];

T = 100;
% we train on many sequences and test on 1
Ntrain = 2;
doSplit = 0;
% labelsTrain{s}(t)
for i=1:Ntrain
  labelsTrain{i} = mc_sample(initStateGen, transMatGen, T);
  labelsTrain01{i} = labelsTrain{i}-1;
end
labelsTest = mc_sample(initStateGen, transMatGen, T);
labelsTest01 = labelsTest-1;

if 0
  D = 2;
  muGen = [1 1; 0 0]';
else
  D = 5;
  muGen = [ones(D,1) 0.5*ones(D,1)];
end
SigmaGen = repmat(eye(D), [1 1 nstates]);

% dataTrain{s}(:,t)
for i=1:Ntrain
  dataTrain{i} = condgauss_sample(muGen, SigmaGen, labelsTrain{i});
  dataTrain1{i} = [dataTrain{i}; ones(1,T)];
end
dataTest = condgauss_sample(muGen, SigmaGen, labelsTest);
dataTest1 = [dataTest; ones(1,T)];


if doSplit
  % split single training sequnce into lots of shorter ones for speed
  Tsmall = 50;
  dataTrain = splitLongSeqIntoManyShort(dataTrain{1}, Tsmall);
  dataTrain1 = splitLongSeqIntoManyShort(dataTrain1{1}, Tsmall);
  labelsTrain = splitLongSeqIntoManyShort(labelsTrain{1}, Tsmall);
  labelsTrain01 = splitLongSeqIntoManyShort(labelsTrain01{1}, Tsmall);
end


% plot data
if 0 % D==2
  data = dataTest; label = labelsTest;
  fh1 = figure;
  set(fh1, 'doublebuffer', 'on')
  for t=1:T
    x = data(1,t); y = data(2,t);
    if label(t)==1
      plot(x, y, 'bo')
    else
      plot(x, y, 'rx')
    end
    hold on
    axis([-4 5 -4 5])
    title(sprintf('t=%d', t))
    pause(0.1)
  end
end




%%%%%%%%%%%%%%%% train/test HMM

[hmm.initState, hmm.transmat, hmm.mu, hmm.Sigma] = ...
    gausshmm_train_observed(dataTrain, labelsTrain, nstates, 'cov_type', 'diag');
hmm.initState = normalize(ones(nstates,1)); % can't estimate prior from 1 sequence!

localEv = mixgauss_prob(dataTest, hmm.mu, hmm.Sigma);
assert(all(localEv(:)>0))
assert(~any(isnan(localEv(:))))
%localEv(:,1:10)

% we do sum-product and take the marginals, not Viterbi decoding
[alpha, beta, gamma, loglik] = fwdback(hmm.initState, hmm.transmat, localEv);
probHMM = gamma(2,:); % state 1= absent,  2 = present
[faRateHMM, dRateHMM] = plotROC(probHMM, labelsTest01);

%%%%%%%%%%% isolated conditional Gaussian

%[iso.mu, iso.Sigma] = mixgaussTrainObserved(dataTrain, labelTrain, nstates);
belGauss = normalize(localEv,1);
probGauss = belGauss(2,:);
[faRateGauss, dRateGauss] = plotROC(probGauss, labelsTest01);


%%%%%%%%%%%%%%%% CRF

addOne = 1;
if addOne
  D1 = D+1;
else
  D1 = D;
end
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
if addOne
  featuresTrainChain = dataTrain1;
  featuresTestChain = dataTest1;
else
  featuresTrainChain = dataTrain;
  featuresTestChain = dataTest;
end

tic
chain = crfchaintrain(chain, featuresTrainChain, labelsTrain, 'gradAlgo', 'scg', ...
		      'checkGrad', 'on', 'MaxIter', maxIter, 'verbose', 1);
toc

belChain = crfchaininfer(chain, featuresTestChain);
probCRF = belChain(2,:);
[faRateCRF, dRateCRF] = plotROC(probCRF, labelsTest01);




%%%%%%%%%%%%%%%% train/test using general CRF code (unrolled to fixed length)

useGeneral = 0; % equivalent results but much slower than chain code

if useGeneral
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
if addOne
  for s=1:Ntrain
    featuresTrainCRF(s,:) = num2cell(dataTrain1{s},1); 
    labelsTrainCRF(s,:) = labelsTrain{s};
  end
  featuresTestCRF = num2cell(dataTest1,1); 
else
  featuresTrainCRF = cell(Ntrain, T);
  for s=1:Ntrain
    featuresTrainCRF(s,:) = num2cell(dataTrain{s},1); 
    labelsTrainCRF(s,:) = labelsTrain{s};
  end
  featuresTestCRF = num2cell(dataTest,1); 
end

seed = 1;
rand('state', seed); randn('state', seed);
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
assert(approxeq(chain.w, net.w{1}))
assert(approxeq(chain.pot, net.pot{1}))

bel = crfinfer(net, featuresTestCRF);
belCRF = cell2num(bel);
assert(approxeq(belChain, belCRF)) % hence ROC is same

end % if general







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
legend('HMM','CRF','logist','Gauss',4)
grid on
axis([0 1 0 1])
title(sprintf('train on %d sequences of length %d in %d dim', ...
	      length(dataTrain), length(dataTrain{1}), D))

if 0
figure;plot(1:T,labelsTest01,'ro-', 1:T,probCRF,'bx-'); title('CRF')
figure;plot(1:T,labelsTest01,'ro-', 1:T,probHMM,'bx-'); title('HMM')
end

drawnow
