function [features1D] = crf2DdemoUncell(features1Dcell,nsamples,nr,nc,D)

features1D = zeros(D, nr*nc, nsamples);
for s=1:nsamples
    i = 1;
    for c=1:nc
        for r=1:nr
            features1D(:,i,s) = features1Dcell{s,i};
            i = i + 1;
        end
    end
end

