function [class2Dtrained] = crf2DdemoClassify(classifier,bel2Dtrained,netTrained,features1Dcell,testNdx,s,nr,nc,nstates)

if strcmp(classifier,'bp')==1 || strcmp(classifier,'gc') == 1
    [Ntest Nvars] = size(features1Dcell(testNdx(s),:));
    if netTrained.addOneToFeatures
        for ss=1:Ntest
            for i=1:Nvars
                features1DcellTemp{s,i} = [features1Dcell{testNdx(s),i};1];
            end
        end
    end
    [localEv, logLocalEv] = crfMkLocalEv(netTrained, features1DcellTemp(s,:)); 
    pot = crfMkPot(netTrained);
    localEvGrid = zeros(nr, nc, nstates);
    i = 1;
    for c=1:nc
        for r=1:nr
            localEvGrid(r,c,:) = localEv{i};
            i = i + 1;
        end
    end
    kernel = pot{1};
    if strcmp(classifier,'gc')==1
        class2Dtrained=graph_cuts(localEvGrid,kernel);
    else
        class2Dtrained=bp_mpe_mrf2_lattice2(kernel, localEvGrid);
    end
    
else
    class2Dtrained = -reshape(bel_to_mpe(reshape(bel2Dtrained,nr*nc,nstates)'),nr,nc)+2;
end