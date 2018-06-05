% Compare different gradient methods for learning a CRF
% on data generated from an HMM (see crfchaindemo).
% Time how long they take to converge,

nstates = 2; % binary classification
initStateGen = [0.5 0.5]';
p = 0.9;
transMatGen = [p 1-p; 1-p p];

T = 100;
Ntrain = 20; % max
% labelsTrain{s}(t)
for i=1:Ntrain
  labelsTrain{i} = mc_sample(initStateGen, transMatGen, T);
  labelsTrain01{i} = labelsTrain{i}-1;
end

D = 10;
muGen = [ones(D,1) 0.5*ones(D,1)];
SigmaGen = repmat(eye(D), [1 1 nstates]);

% dataTrain{s}(:,t)
for i=1:Ntrain
  dataTrain{i} = condgauss_sample(muGen, SigmaGen, labelsTrain{i});
end
featuresTrainChain = dataTrain;

chain = crfchain(D, nstates);
w1 = chain.w;
pot1 = chain.pot;

% quasinew often has numerical problems (bug in netlab?)
% conjgrad is very slow
% fminunc has fewer calls to the fn/grad, but on large problems is slower than scg overall.

%algos = {'quasinew', 'scg', 'fminunc', 'conjgrad'};
algos = {'scg', 'fminunc'};
clear time trainedNet finalErr nfn ngrad out
maxIter = 100; 
for a=1:length(algos)
  tic
  chain.w = w1;
  chain.pot = pot1;
  [trainedNet{a}, finalErr(a), nfn(a), ngrad(a), out{a}] = ...
      crfchaintrain(chain, featuresTrainChain, labelsTrain, ...
		    'gradAlgo', algos{a}, ...
		    'checkGrad', 'off', 'MaxIter', maxIter, 'verbose', 1);
  time(a)=toc
end
cost = nfn + ngrad

if 0
  % The methods are suppsoed to reach the global optimum (if run to convergence)
  % but due to different convergence criteria, this is not always exact.
  tol = 1e-1;
  for a=2:length(algos)
    assert(approxeq(trainedNet{a}.w, trainedNet{1}.w, tol))
    assert(approxeq(trainedNet{a}.pot, trainedNet{1}.pot, tol))
  end
end

