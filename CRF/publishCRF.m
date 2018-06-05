% Copy over source files to one location
% so they can be zipped and released to others

Base = 'C:\kmurphy\matlab\Wearables\UBCcode';

target = 'C:\kmurphy\matlab\CRFall';

mkdirKPM(target)

% We don't need to copy over subdirectories,
% because cp -r will take care of that
%	   'BPlattice2\Lattice2',  'BPlattice2\MPI',...



folders = {'CRF', 'CRF\CRF1D', 'CRF\CRF2D',...
	   'BPMRF2',  'BPlattice2',...
	   'BPlattice2\Lattice2',  'BPlattice2\MPI',...
	   'HMM', 'KPMtools', 'KPMstats', 'graph', ...
	   'Foreign\netlab\netlab3.3'};

curdir = pwd;

for f=1:length(folders)
  src = fullfile(Base, folders{f});
  dst = fullfile(target, folders{f});
  
  % copy source directory to destination directory
  if 1 % ~isempty(strfind(folders{f}, '\'))
    mkdir(dst)
  end
  %cmd = sprintf('cp -r %s %s', src, dst)
  %cmd  = sprintf('cd %s; cp *.* %s', src, dst)
  %cmd  = sprintf('cp %s\\*.*  %s', src, dst)
  cd(src)
  cmd = sprintf('cp *.* %s',  dst)
  system(cmd)

  dstCVS = fullfile(dst, 'CVS');
  cmd = sprintf('rm -rf  %s', dstCVS)
  system(cmd)
end

fname = fullfile(target, sprintf('created-%s.mat', date));
dummy = rand(1,1);
save(fname, 'dummy')

cd(curdir)
