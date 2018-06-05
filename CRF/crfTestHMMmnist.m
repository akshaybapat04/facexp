% Generate data from an HMM with Gaussian outputs
% By making the centers closer, the classification problem becomes harder.
% But we should be able to exploit the temporal correlation.

seed = 0;
rand('state', seed); randn('state', seed);

nstates = 2; % binary classification
initStateGen = [0.5 0.5]';
transMatGen = [0.8 0.2; 0.2 0.8];

T = 200;
Nex = 1; % we use a single long sequence
labelsTrain = mc_sample(initStateGen, transMatGen, T, Nex);
labelsTest = mc_sample(initStateGen, transMatGen, T, Nex);
labelsTest01 = labelsTest-1;
labelsTrain01 = labelsTrain-1;


useMNIST = 1;

if useMNIST
  load('mnist.mat');
  crop = 4; % we ignore the border pixels, which are usually all off
  subsample = 4; % we subsample the data for speed
  
  [nr nc ndata] = size(mnist.train_images);
  dataTrain = zeros(nr, nc, T);
  digits = [1 0]; % mapping from HMM states to digit numbers
  for q=1:nstates
    ndxDest = find(labelsTrain==q);
    ndxSrc = find(mnist.train_labels==digits(q));
    dataTrain(:,:,ndxDest) = mnist.train_images(:,:,ndxSrc(1:length(ndxDest)));
  end
  dataTrain = dataTrain(crop+1:end-crop, crop+1:end-crop, :);
  dataTrain = dataTrain(1:subsample:end, 1:subsample:end, :);
  dataTrain = reshape(dataTrain, size(dataTrain,1)*size(dataTrain,2), T);
  [dataTrain, stndMu, stndVar] = standardize(dataTrain);
  dataTrain = dataTrain(3:end,:); % remove all 0s pixels
  dataTrain = mk_unit_norm(dataTrain);
  
  [nr nc ndata] = size(mnist.test_images);
  dataTest = zeros(nr, nc, T);
  digits = [1 0]; % mapping from HMM states to digit numbers
  for q=1:nstates
    ndxDest = find(labelsTest==q);
    ndxSrc = find(mnist.test_labels==digits(q));
    dataTest(:,:,ndxDest) = mnist.test_images(:,:,ndxSrc(1:length(ndxDest)));
  end
  dataTest = dataTest(crop+1:end-crop, crop+1:end-crop, :);
  dataTest = dataTest(1:subsample:end, 1:subsample:end, :);
  dataTest = reshape(dataTest, size(dataTest,1)*size(dataTest,2), T);
  dataTest = standardize(dataTest, stndMu, stndVar);
  dataTest = dataTest(3:end,:); % remove all 0s pixels
  dataTest = mk_unit_norm(dataTest);

  D = size(dataTrain,1);
else
  if 1
    D = 2;
    muGen = [1 1; 0 0]';
  else
    D = 5;
    muGen = [ones(D,1) 0.5*ones(D,1)];
  end
  SigmaGen = repmat(eye(D), [1 1 nstates]);
  dataTrain = mixgauss_sample(muGen, SigmaGen, labelsTrain);
  dataTest = mixgauss_sample(muGen, SigmaGen, labelsTest);

  if 0 % D==2
    data = dataTest; label = labelTest;
    fh1 = figure;
    set(fh1, 'doublebuffer', 'on')
    for s=1:1 %Nex
      for t=1:T
	x = data(1,t,s); y = data(2,t,s);
	if label(t,s)==1
	  plot(x, y, 'bo')
	else
	  plot(x, y, 'rx')
	end
	hold on
	axis([-4 5 -4 5])
	title(sprintf('s=%d t=%d', s, t))
	pause(0.1)
      end
    end
  end

end



%%%%%%%%%%%%%%%% train/test HMM

[hmm.initState, hmm.transmat, hmm.mu, hmm.Sigma] = ...
    gausshmm_train_observed(dataTrain, labelsTrain, nstates, 'cov_type', 'diag');
hmm.initState = normalize(ones(nstates,1)); % can't estimate prior from 1 sequence!

localEv = mixgauss_prob(dataTest, hmm.mu, hmm.Sigma);
assert(all(localEv(:)>0))
assert(~any(isnan(localEv(:))))
localEv(:,1:10)

% we do sum-product and take the marginals, not Viterbi decoding
[alpha, beta, gamma, loglik] = fwdback(hmm.initState, hmm.transmat, localEv);
probHMM = gamma(2,:); % state 1= absent,  2 = present
[faRateHMM, dRateHMM] = plotROC(probHMM, labelsTest01);

%%%%%%%%%%% isolated conditional Gaussian

%[iso.mu, iso.Sigma] = mixgaussTrainObserved(dataTrain, labelTrain, nstates);
belGauss = normalize(localEv,1);
probGauss = belGauss(2,:);
[faRateGauss, dRateGauss] = plotROC(probGauss, labelsTest01);

%%%%%%%%%%%%%%%% train/test CRF using general code

G = diag(ones(T-1,1),1); 
%G = zeros(T,T);
%for t=1:T-1
%  G(t,t+1)=1;
%end
regularizer = 1;
net = crf(repmat(D+1,1,T), repmat(nstates,1,T), G, 'eclassNode', ones(1,T), ...
	  'eclassEdge', ones(1,T-1), 'clampWeightsForOneState', 1, ...
	  'alpha', regularizer);

dataTrain1 = [dataTrain; ones(1,T)];
featuresTrain = num2cell(dataTrain1,1); % featuresTrain{t}(:)
net = crftrain(net, featuresTrain, labelsTrain(:)', 'gradAlgo', 'scg', 'checkGrad', 'on');

dataTest1 = [dataTest; ones(1,T)];
featuresTest = num2cell(dataTest1,1); % featuresTest{t}(:)
bel = crfinfer(net, featuresTest);
belM = cell2num(bel);
probCRF = belM(2,:);
[faRateCRF, dRateCRF] = plotROC(probCRF, labelsTest01);


%%%%%%%%%%%%%% fit isolated logistic classifier

if useMNIST
  % Hessian is singular!
  faRateLogist = zeros(size(faRateCRF));
  dRateLogist = zeros(size(dRateCRF));
else
  beta = logist2Fit(labelsTrain01, dataTrain1);
  probLogist = logist2Apply(beta, dataTest1);
  [faRateLogist, dRateLogist] = plotROC(probLogist, labelsTest01);
end

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

figure;plot(1:T,labelsTest01,'ro-', 1:T,probCRF,'bx-')
figure;plot(1:T,labelsTest01,'ro-', 1:T,probHMM,'bx-')


