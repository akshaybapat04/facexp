% Learn a 2D CRF using general CRF code plus hmm-lattice inference

clear all
p = 0.75;
kernel = [p 1-p; 1-p p];
doLearning = 1;

nr = 7;
nc = 7;
Nnodes = nr*nc;
nrows = nr;
ncols = nc;
Nedges = (nrows-1)*ncols + nrows*(ncols-1);
nstates = 2;
G = mk_2D_lattice(nr, nc, 4);
%G = zeros(Nnodes, Nnodes);
D = 1; % scalar observations

seed = 1;
rand('state', seed); randn('state', seed);
eclassEdge = ones(1,Nedges);
adjustableEdge = 1;
netOrig = crf(repmat(D,1,Nnodes), repmat(nstates,1,Nnodes), G, 'eclassNode', ones(1,Nnodes), ...
	  'eclassEdge', eclassEdge, 'alpha', 0, 'clampWeightsForOneState', 1, ...
	  'addOneToFeatures', 1, 'adjustableEdgeEclassBitv', [adjustableEdge]);
w1 = netOrig.w{1};
pot1 = netOrig.pot{1};
if ~adjustableEdge
  netOrig.pot{1} = kernel; % set params to truth
end

netOrig.infEngineName = 'lattice2hmmCell'; % crfgrad calls lattice2hmmCellInfer
netOrig.infEngine = lattice2hmmCellEngine(nr, nc, nstates);

%%%%%%%%%%% Training


mu = [120 130];
Sigma = ones(1,1,2);
Sigma(:,:,1) = 50;
Sigma(:,:,2) = 50;

nsamples = 20;
[labels, obs] = lattice2_hmm_sample(kernel, nr, nc, nsamples, mu, Sigma);

trainNdx = 1:10;
testNdx = 11:15;

% In this case, the features happen to be scalar.
% features1Dcell{s,i}(:)
% features1D(:,i,s)
% labels1D(s,i)
features1Dcell = cell(nsamples, Nnodes);
features1D = zeros(D, Nnodes, nsamples);
labels1D = zeros(nsamples, Nnodes);
for s=1:nsamples
  i = 1;
  for c=1:nc
    for r=1:nr
      features1Dcell{s,i} = obs(r,c,s);
      features1D(:,i,s) = obs(r,c,s);
      labels1D(s,i) = labels(r,c,s);
      i = i + 1;
    end
  end
  
  if 1
  figure(1); imagesc(labels(:,:,s)); colorbar
  figure(2); image(obs(:,:,s)); colormap(gray(256)); colorbar
  drawnow
  %pause
  end
end


if doLearning
maxIter = 50;
tic
netTrained = crftrain(netOrig, features1Dcell(trainNdx,:), labels1D(trainNdx,:), ...
	       'gradAlgo', 'fminunc', ...
	       'checkGrad', 'off', 'MaxIter', maxIter);
toc
end

% trainData(:,is)
lab = -(labels1D(trainNdx,:)-2); % map 1->1, 2->0 (absent)
trainData = reshape(features1D(:,:,trainNdx), [D Nnodes*length(trainNdx)]);
logistW = logist2Fit(lab(:), trainData);

netClampedAll = netOrig;
netClampedAll.pot{1} = kernel;
netClampedAll.w{1} = logistW;


%%%%%%%% Testing

labelsTest = [];
probLogistTest = [];
probTrainedTest = [];
probClampedAllTest = [];

Ntest = length(testNdx);
for s=1:Ntest
  testData = features1D(:,:,testNdx(s)); % testData(:,i)
  probLogist = logist2Apply(logistW, testData);
  
  if doLearning
    belCellTrained = crfinfer(netTrained, features1Dcell(testNdx(s), :));
    bel2Dtrained = zeros(nr, nc, nstates);
  end
  belCellClampedAll = crfinfer(netClampedAll, features1Dcell(testNdx(s), :));
  bel2DclampedAll = zeros(nr, nc, nstates);
  bel2Dlogist = zeros(nr, nc, nstates);
  labels2D = zeros(nr, nc);
  i = 1;
  for c=1:nc
    for r=1:nr
      if doLearning
	bel2Dtrained(r,c,:) = belCellTrained{i};
      end
      bel2DclampedAll(r,c,:) = belCellClampedAll{i};
      bel2Dlogist(r,c,:) = [probLogist(i); 1-probLogist(i)];
      labels2D(r,c) = -(labels1D(testNdx(s), i)-2); % map 1->1, 2->0
      i = i + 1;
    end
  end
  labelsTest = [labelsTest; labels2D(:)];
  probLogistTest = [probLogistTest; probLogist(:)];
  if doLearning
    tmp = bel2Dtrained(:,:,1); probTrainedTest = [probTrainedTest; tmp(:)];
  end
  tmp = bel2DclampedAll(:,:,2); probClampedAllTest = [probClampedAllTest; tmp(:)];
  
  figure(1); clf; imagesc(labels2D); title('truth'); colormap(gray(256)); colorbar
  if doLearning
    figure(2); clf; imagesc(bel2Dtrained(:,:,1)); title('crf trained'); colormap(gray(256)); colorbar
  end
  figure(3); clf; imagesc(bel2DclampedAll(:,:,2)); title('crf clamped all'); colormap(gray(256)); colorbar
  figure(4); clf; imagesc(bel2Dlogist(:,:,1)); title('logist'); colormap(gray(256)); colorbar
  drawnow
  pause
end

[faRateLogist, dRateLogist] = plotROC(probLogistTest, labelsTest);
if doLearning
  [faRateTrained, dRateTrained] = plotROC(probTrainedTest, labelsTest);
end
[faRateClampedAll, dRateClampedAll] = plotROC(probClampedAllTest, labelsTest);
%[faRateClampedAll1, dRateClampedAll1] = plotROC(1-probClampedAllTest, labelsTest);
figure; hold on
plot(faRateLogist, dRateLogist, 'ro-');
if doLearning
  plot(faRateTrained, dRateTrained, 'bx-');
end
plot(faRateClampedAll, dRateClampedAll, 'gs-');
%plot(faRateClampedAll1, dRateClampedAll1, 'kd-');
if doLearning
  legend('logist','trained','clamped')
else
  legend('logist','clamped')
end
grid on
axis([0 1 0 1])
