folders = {'CRF', 'BPMRF2', 'HMM', 'KPMtools', 'KPMstats', 'graph', ...
	  'Foreign\netlab\netlab3.3'};


Base = 'C:/kmurphy/matlab';
if ~exist(fullfile(Base, 'CRFall'), 'dir')
  %[root, directory, ext, ver] = fileparts(Base);
  mkdir(Base, 'CRFall')
else
  rmdir(fullfile(Base, 'CRFall'))
  mkdir(Base, 'CRFall')
end

for f=1:length(folders)
  src = fullfile(Base, folders{f});
  dest = fullfile(Base, 'CRFall', folders{f});
  cmd = sprintf('xcopy  %s %s /s /i', src, dest);
  cmd
  system(cmd)
end
