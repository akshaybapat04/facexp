clc;
Base = 'C:\Users\aksha_o463x8r\Downloads\CRFall\CRFall';

folders = {'CRF', 'CRF\CRF1D', 'CRF\CRF2D', ...
	   'BPMRF2',  'BPlattice2', 'BPlattice2\Lattice2',  'BPlattice2\MPI',...
	   'HMM', 'KPMtools', 'KPMstats', 'graph', ...
	  'Foreign/netlab/netlab3.3'};

for f=1:length(folders)
  addpath(fullfile(Base, folders{f}))
end
