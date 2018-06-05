function [infEngine] = crf2DdemoInfEngine(infEngineName,nr,nc,nstates,netOrig);
% A wrapper function since the different inference engines take different
% parameters

if strcmp(infEngineName,'lattice2hmmCell')==1
    infEngine = lattice2hmmCellEngine(nr,nc,nstates);
elseif strcmp(infEngineName,'bp_mrf2_lattice2')==1
    infEngine = bp_mrf2_lattice2Engine(netOrig.E,netOrig.G,nr,nc,nstates,50);
else
    infEngine = bploopyEngine(netOrig.E,netOrig.nstates,'max_iter',50);
end
