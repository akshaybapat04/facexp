function [net, finalErr, nfn, ngrad, options]  = crfchaintrain(net, features, labels, varargin)
% CRFTRAIN Find globally optimal maximum likelihood parameter estimates from fully labeled data
% [net, finalErr, nfn, ngrad, options]  = crfchaintrain(net, features, labels,...)
%
% features{s}(:,t) - cell for time t case s is a column feature vector
% labels{s}(t) - desired discrete labels (targets) for time t, case s
%
% Optional arguments
% gradAlgo - fminunc (optimization toolbox) large scale trust-region
%          - scg (netlab) scaled conjugate gradient
%          - quasinew (netlab) quasi-Newton: faster but often numerically unstable

[options, maxIter, errTol, paramTol, checkGrad, gradAlgo, verbose, multiStage] = process_options(...
    varargin, 'options', [], 'maxIter', 100, 'errTol', 1e-2, ...
    'paramTol', 1e-2, 'checkGrad', 'off', 'gradAlgo', 'scg', 'verbose', 0, ...
    'multiStage', 0);


if isempty(options)
  if strcmpi(gradAlgo, 'fminunc') % matlab
    options = optimset('GradObj', 'on', 'Diagnostics', 'on', 'Display', 'iter', ...
		       'MaxIter', maxIter, 'TolFun',  errTol, 'DerivativeCheck', checkGrad);
  else % netlab
    options = foptions;
    options(1) = verbose;
    options(2) = paramTol;
    options(3) = errTol;
    options(9) = strcmpi(checkGrad, 'on');
    options(14) = maxIter;
  end
end

fprintf('crfchaintrain with %d training cases\n', length(features));
switch gradAlgo
 case 'fminunc',
  w = crfchainpak(net);
  [w, finalErr, exitFlag, out] = fminunc(@crfchainErrAndGrad, w, options, net, features, labels);
  net = crfchainunpak(net, w);
  nfn = out.funcCount;
  ngrad = out.cgiterations;
  options = out;
 case 'minimize',
  disp('Calling minimize');
  w = crfchainpak(net)';
  
  [w, fX, ngrad] =  minimize(w, 'crfchainMinimizeErrAndGrad', maxIter, net, features, labels);
  net = crfchainunpak(net, w');
 case 'olgd',
  [net, options, errlog] = olgd(net, options, features, labels);
  finalErr = options(8);
  nfn = options(10);
  ngrad = options(11);
 otherwise,
  [net, options] = netopt(net, options, features, labels, gradAlgo);
  finalErr = options(8);
  nfn = options(10);
  ngrad = options(11);
end

