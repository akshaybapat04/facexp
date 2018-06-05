function crf2Ddemo(varargin)
% An example of building, training, and doing inference/classification
% in 2D lattice CRFs
% 
% optional Arguments:
%
% doDisplay 
%           - 0 only display results (default)
%           - 1 display training/testing examples before learning
% infEngineName
%           - lattice2hmmCell exact inference by converting to an HMM
%           (default)
%           - bploopy approximate inference with BP
%           - bp_mrf2_lattice approximate inference with vectorized BP
% method 
%           - QN-TR (optimization toolbox): Quasi-Newton Trust-region
%           - QN-LS (optimization toolbox): Quasi-Newton Linesearch
%           (default)
%           - N-TR (optimization toolbox): Newton Trust-region
%           - scg (netlab) scaled conjugate gradient
%           - minimize: Carl Rasmussen's minimize.m
%           - N-LS: Michael Friedlander's Newton Linesearch
% data
%           - exact generate data from a potential model by converting to
%           HMM
%           - digits digit segmentation (16 x 16)
%           - lines binary denoising (32 x 32)
% classify
%           - '' output beliefs
%           - hmm exact classification using HMM beliefs
%           - bp approximate classification using BP beliefs
%           - gc exact classification in binary models with graph cuts
% maxIter
%           - [ positive intger ] (default = 50)

% Process script arguments and assign defaults
[doDisplay,doLearning,infEngineName,method,data,classifier,maxIter] = ...
    process_options(varargin,'doDisplay',0,'doLearning',1,....
    'infEngineName','lattice2hmmCell','method','QN-LS',...
    'data','exact','classifier','','maxIter',50);

% Set the indices of training and test examples, 
% the number of states, and the number of examples to generate/load

trainNdx = 1:3;
testNdx = 3:6;
nstates = 2;
nsamples = 6;

% Load one of the three toy data sets into 1D structures used by the code
% In this case, the features happen to be scalar.
% features1Dcell{sample,i}(:)
% labels1D(sample,i)
% nr, nc: Number of rows and columns in lattice models

[labels1D,features1Dcell,nr,nc,D]=crf2DdemoGetData(data,nsamples,doDisplay);
Nnodes = nr*nc;

% Create and (randomly) initialize a CRF model
% The local potentials are stored in netOrig.w
% The edge potentials are stored in netOrig.pot

netOrig = crf2DdemoInitialize(nr,nc,nstates);

% Set the inference engine
% This is used for evaluating the objective and gradient

netOrig.infEngineName = infEngineName;
netOrig.infEngine = crf2DdemoInfEngine(infEngineName,nr,nc,nstates,netOrig);

% ********** TRAINING ***************************************

if doLearning
    tic
    netTrained = crf2DdemoTrain(netOrig, features1Dcell(trainNdx,:), labels1D(trainNdx,:), ...
        'gradAlgo', method, ...
        'MaxIter', maxIter);
    toc
end

% As a comparison, train a Logistic Regression Classifier
lab = -(labels1D(trainNdx,:)-2);
features1D = crf2DdemoUncell(features1Dcell,nsamples,nr,nc,D);
trainData = reshape(features1D(:,:,trainNdx), [D Nnodes*length(trainNdx)]);
logistW = logist2Fit(lab(:), trainData);


% **************** Testing ***********************************

Ntest = length(testNdx);
for s=1:Ntest
    
    % Compute the logistic regression probabilities
    
    probLogist = logist2Apply(logistW, features1D(:,:,testNdx(s)));

    % Compute the CRF beliefs
    
    if doLearning
        belCellTrained = crfinfer(netTrained, features1Dcell(testNdx(s), :));
        bel2Dtrained = zeros(nr, nc, nstates);    
    end
    
    % Convert to a 2D Image Representation
    bel2Dlogist = zeros(nr, nc, nstates);
    labels2D = zeros(nr, nc);
    i = 1;
    for c=1:nc
        for r=1:nr
            obs2D(r,c,:) = features1Dcell{testNdx(s),i};
            labels2D(r,c) = labels1D(testNdx(s), i); 
            
            bel2Dlogist(r,c,:) = [ 1-probLogist(i);probLogist(i);];
            
            if doLearning
                bel2Dtrained(r,c,:) = [belCellTrained{i}(2) belCellTrained{i}(1)];
            end 
            
            i = i + 1;
        end
    end
    
    
    % Now classify the images
    doClassify = 0;
    if strcmp(classifier,'') ~= 1
        doClassify = 1;
        classLogist = -reshape(bel_to_mpe(reshape(bel2Dlogist,nr*nc,nstates)'),nr,nc)+2;
        if doLearning
            class2Dtrained=crf2DdemoClassify(classifier,bel2Dtrained,netTrained,features1Dcell,testNdx,s,nr,nc,nstates);
        end
    end
    
    % Display the test results
    figure(1); clf; imagesc(obs2D); title('test instace'); colormap(gray(256)); colorbar
    figure(2); clf; imagesc(labels2D); title('truth'); colormap(gray(256)); colorbar
    if doLearning
        if doClassify
            figure(4); clf; imagesc(class2Dtrained(:,:,1));title('crf trained Classified'); colormap(gray(256)); colorbar
        else
            figure(4); clf; imagesc(bel2Dtrained(:,:,1)); title('crf trained'); colormap(gray(256)); colorbar
        end
    end
    if doDisplay
        if doClassify
            figure(3); clf; imagesc(classLogist(:,:,1));title('logistClassified'); colormap(gray(256)); colorbar
        else
            figure(3); clf; imagesc(bel2Dlogist(:,:,1)); title('logist'); colormap(gray(256)); colorbar
        end
    end
    
    drawnow
    pause
end
