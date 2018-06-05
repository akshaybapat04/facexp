function [net, finalErr, nfn, ngrad, options]  = crftrain(net, x, t, varargin)
% CRFTRAIN Find globally optimal maximum likelihood parameter estimates from fully labeled data
% function [net, finalErr, nfn, ngrad, options]  = crftrain(net, x, t, ...)
%
% x{s,i}(:) = feature vector for node i case s
% t(s,i) = target value (in 1..numStates(i)) for node i in case s
%
% Optional arguments
% gradAlgo - QN-TR (optimization toolbox): Quasi-Newton Trust-region
%          - QN-LS (optimization toolbox): Quasi-Newton Linesearch
%          - N-TR (optimization toolbox): Newton Trust-region
%          - scg (netlab) scaled conjugate gradient
%          - minimize: Carl Rasmussen's minimize.m
%          - N-LS: Michael Friedlander's Newton Linesearch



if any(t(:)==0)
    error('targets must be in range 1,2,...; zero not allowed')
end
[options, maxIter, errTol, paramTol, checkGrad, gradAlgo, verbose, multiStage,LargeScale,GradObj] = process_options(...
    varargin, 'options', [], 'maxIter', 50, 'errTol', 1e-5, ...
    'paramTol', 1e-2, 'checkGrad', 'off', 'gradAlgo', 'QN-LS', 'verbose', 1, ...
    'multiStage', 0,'LargeScale','on','GradObj','on');

if isempty(options)
    if strcmpi(gradAlgo, 'scg')% netlab
        options = foptions;
        options(1) = verbose;
        options(2) = paramTol;
        options(3) = errTol;
        options(9) = strcmpi(checkGrad, 'on');
        options(14) = maxIter;
    else
        options = optimset('GradObj', GradObj, 'Diagnostics', 'on', 'Display', 'iter', ...
            'MaxIter', maxIter, 'TolFun',  errTol, 'DerivativeCheck', checkGrad);
    end
end

[Ntrain Nvars] = size(x);
if net.addOneToFeatures
    for s=1:Ntrain
        for i=1:Nvars
            x{s,i} = [x{s,i};1];
        end
    end
end


% it's faster to train with a subset of the data to get a good initial estimate
% and then retrain with all the data (two stages) than to use all the data
% at once
if multiStage 
    pi1 = randperm(Ntrain); pi2 = randperm(Ntrain); pi3 = randperm(Ntrain);
    %ndx = {1:10:Ntrain, 1:1:Ntrain};
    % We don't ever want to train with very small datasets (say < 50)
    % in case we encounter numericla problems.
    if strcmpi(gradAlgo, 'scg')
        minSize = 50;
    else
        minSize = 100;
    end
    ndx = {};
    if Ntrain>=minSize*10
        ndx(end+1) = {pi1(1:round(Ntrain/10))};
    elseif Ntrain>=minSize*5
        ndx(end+1) = {pi2(1:round(Ntrain/5))};
    elseif Ntrain>=minSize*2
        ndx(end+1) = {pi3(1:round(Ntrain/2))};
    end
    ndx{end+1} = 1:Ntrain;
else
    ndx = {1:Ntrain};
end

% Now the actual training
for i=1:length(ndx)
    fprintf('crftrain with %d training cases\n', length(ndx{i}));
    switch gradAlgo
        case 'QN-TR'
            w = crfpak(net);
            options.LargeScale = 'on';
            [w, finalErr, exitFlag, out] = fminunc(@crfErrAndGrad, w, options, net, x(ndx{i},:), t(ndx{i},:));
            net = crfunpak(net, w);
            nfn = out.funcCount
            options = out;
        case 'QN-LS'
            w = crfpak(net);
            options.LargeScale = 'off';
            [w, finalErr, exitFlag, out] = fminunc(@crfErrAndGrad, w, options, net, x(ndx{i},:), t(ndx{i},:));
                        fprintf('Done training\n');
            net = crfunpak(net, w);
            nfn = out.funcCount
        case 'N-TR'
            fprintf('This case not implemented yet...\n');
            assert(0==1);
            w = crfpak(net);
            options.Hessian = 'on';
            [w, finalErr, exitFlag, out] = fminunc(@crfErrAndGradAndHess, w, options, net, x(ndx{i},:), t(ndx{i},:));
            net = crfunpak(net, w);
            nfn = out.funcCount
        case 'minimize'
            w = crfpak(net);
            [w,fW,nfn] = minimize_ras(w,@crfErrAndGrad,maxIter,net,x(ndx{i},:),t(ndx{i},:));
            net = crfunpak(net,w);
        case 'fminsearch',
            w = crfpak(net);
            [w, finalErr, exitFlag, out] = fminsearch(@crfErrAndGrad, w, options, net, x(ndx{i},:), t(ndx{i},:));
            net = crfunpak(net, w);
            nfn = out.funcCount;
        case 'N-LS'
            fprintf('This case not implemented yet...\n');
            assert(0==1);
            w = crfpak(net);
            w = newton_min(@crfErrAndGradAndHess,w,'newton',0,1e-7,100,net,x(ndx{i},:),t(ndx{i},:));
            net = crfunpak(net,w);
        otherwise,
            [net, options] = netopt(net, options, x(ndx{i},:), t(ndx{i},:), gradAlgo);
            finalErr = options(8);
            nfn = options(10);
            ngrad = options(11);
    end
end

