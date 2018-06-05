% see how accurate loopy BP is on random graphs with binary states


clear all

%seed = 0;
%rand('state', seed);

nnodes = 9;
nstates = 2*ones(1,nnodes);
ntrials = 10;

for trial=1:ntrials

  if 0
    [G, pot, localEv] = mkRndMRF(nnodes, 100);
  else
    localEvSig = 0.1; 
    potSig = 0.1;
    %[G, pot, localEv, J] = spinGlass(sqrt(nnodes), localEvSig, potSig);
    [G, pot, localEv, J] = spinGlass(3, 4, 0.2); % hard to solve by BP
  end
  
  [belExactCell] = brute_force_inf_mrf2(G, pot, nstates, localEv);
  belExact = cell2num(belExactCell);
  
  [E, Nedges] = assignEdgeNums(G);
  engine = bploopyEngine(E, nstates);
  [belBPCell] = bploopyInfer(engine, pot, localEv);
  belBP = cell2num(belBPCell);
  
  figure(2); clf
  plot(belExact(1,:), 'ro-')
  hold on
  plot(belBP(1,:), 'bx-')
  drawnow
  
  err(trial) = sum(abs(belExact(1,:)-belBP(1,:)))/nnodes;
  
end % next trial

mean(err,2)

