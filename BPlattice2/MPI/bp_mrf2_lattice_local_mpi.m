% Script for doing bp_mrf_lattice_local in parallel.

MPI_Init;
% Set who is leader.
leader = 0;

% Get size and rank.
comm_size = MPI_Comm_size(MPI_COMM_WORLD);
my_rank = MPI_Comm_rank(MPI_COMM_WORLD);
disp(['rank: ' num2str(my_rank)]);

% Setup tags.
ITERATION = 0;
EVIDENCE_TAG = 1000000;
PARAMETERS_TAG = 2000000;
EAST_MSGS_TAG = 3000000;
WEST_MSGS_TAG = 4000000;
OUTPUT_TAG = 5000000;

% Compute current message tags.
parameters_tag_i = PARAMETERS_TAG + ITERATION;
evidence_tag_i = EVIDENCE_TAG + ITERATION;
east_msgs_tag_i = EAST_MSGS_TAG + ITERATION;
west_msgs_tag_i = WEST_MSGS_TAG + ITERATION;
output_tag_i = OUTPUT_TAG + ITERATION;

% Leader.
if (my_rank == leader)

  % local_ev = zeros(nrows, ncols_per_strip, nstates, nstrips);
  local_ev = zeros(nrows, ncols_per_strip, nstates);


  % Send data to each processor.
  if (comm_size > 1)

    % Broadcast parameters.
    MPI_Bcast(my_rank,parameters_tag_i,MPI_COMM_WORLD, ...
      pot,nstrips, max_iter, momentum, tol, maximize, verbose, ...
      nrows,ncols,ncols_per_strip,ndir,nstates);

    % Distribute local part of local evidence.
    for i = 2:comm_size
      s = i;
      scols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip);
      local_ev(:,:,:) = local_evidence(:, scols, :);
      i_rank = i - 1;
      MPI_Send(i_rank,evidence_tag_i,MPI_COMM_WORLD,local_ev);
    end

  end

  % Handle rank=0 processor.
  s = 1;
  scols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip);
  local_ev(:,:,:) = local_evidence(:, scols, :);


end

if (my_rank ~= leader)

  % Receive global data.
  [pot,nstrips, max_iter, momentum, tol, maximize, verbose, ...
   nrows,ncols,ncols_per_strip,ndir,nstates] ...
   = MPI_Recv(leader,parameters_tag_i,MPI_COMM_WORLD);

  % Receive local data.
  [local_ev] = MPI_Recv(leader,evidence_tag_i,MPI_COMM_WORLD);

end

% Now everyone should have the same state information and
% can proceed with the calculation.

% initialise local beliefs
prod_of_msgs = local_ev;
old_bel = local_ev;
new_bel = old_bel;

% msgs(r,c,dir,:,s) is the message sent *into* (r,c) from direction dir for strip s
% where dir=1 comes from west, 3 from east, 2 from south, 4 from north.

%old_msgs = normalize(ones(nrows, ncols_per_strip, ndir, nstates, nstrips), 4);
old_msgs = normalize(ones(nrows, ncols_per_strip, ndir, nstates), 4);
msgs = old_msgs;

% We add dummy rows and columns on the boudnaries which always send msgs of all 1s
% and which never get updated. This simplifies the code.

% Messages on dummy boundaries are all 1s so as not to effect multiplications.
% Since we divide the interior into vertical strips, 
% the internal vertical boundaries will later be changed.
% The horizontal boundary messages are always clamped at 1, and do not need to be stored.


%old_msgs_east = ones(nrows, ndir, nstates, nstrips);
%old_msgs_west = ones(nrows, ndir, nstates, nstrips);
%prod_of_msgs_east = ones(nrows, nstates, nstrips);
%prod_of_msgs_west = ones(nrows, nstates, nstrips);


old_msgs_east = ones(nrows, ndir, nstates);
old_msgs_west = ones(nrows, ndir, nstates);
prod_of_msgs_east = ones(nrows, nstates);
prod_of_msgs_west = ones(nrows, nstates);

converged = 0;
iter = 1;

MAX_ITERATION = 2;

% while ~converged & (iter <= max_iter)

for ITERATION=1:MAX_ITERATION

disp(['ITERATION: ' num2str(ITERATION)]);

  % Compute current message tags.
  parameters_tag_i = PARAMETERS_TAG + ITERATION;
  evidence_tag_i = EVIDENCE_TAG + ITERATION;
  east_msgs_tag_i = EAST_MSGS_TAG + ITERATION;
  west_msgs_tag_i = WEST_MSGS_TAG + ITERATION;
  output_tag_i = OUTPUT_TAG + ITERATION;

  % copy msgs in interior into old_msgs before modifying 
  %for s=1:nstrips
    %get rid of zero terms, since we'll be dividing
    %old_msgs(:,:,:,:,s) = msgs(:,:,:,:,s) + (msgs(:,:,:,:,s) ==0);
    old_msgs(:,:,:,:) = msgs(:,:,:,:) + (msgs(:,:,:,:) ==0);
  %end
  
  % Send/Receive to/from your neighbor.
  if (comm_size > 1)
    s = my_rank+1;
    east_rank = my_rank+1;
    west_rank = my_rank-1;
    if s>1
      MPI_Send(west_rank,west_msgs_tag_i,MPI_COMM_WORLD, ...
        squeeze(old_msgs(:,1,:,:)),squeeze(prod_of_msgs(:,1,:)));
    end
    if s<nstrips
      MPI_Send(east_rank,east_msgs_tag_i,MPI_COMM_WORLD, ...
        squeeze(old_msgs(:,end,:,:)),squeeze(prod_of_msgs(:,end,:)));
    end
    if s>1
%      [old_msgs_west(:,:,:),prod_of_msgs_west(:,:)] = ...
      [old_msgs_east(:,:,:),prod_of_msgs_east(:,:)] = ...
        MPI_Recv(west_rank,east_msgs_tag_i,MPI_COMM_WORLD);
    end
    if s<nstrips
%      [old_msgs_east(:,:,:),prod_of_msgs_east(:,:)] = ...
      [old_msgs_west(:,:,:),prod_of_msgs_west(:,:)] = ...
        MPI_Recv(east_rank,west_msgs_tag_i,MPI_COMM_WORLD);
    end
  end

%  if (ITERATION==1)
%    save(['parallel' num2str(my_rank)])
%  end

  % compute new msgs
  %for s=1:nstrips
    %msgs(:,:,:,:,s) = comp_msgs(old_msgs(:,:,:,:,s), prod_of_msgs(:,:,:,s), ...
    %				    old_msgs_east(:,:,:,s), old_msgs_west(:,:,:,s),...
    %				    prod_of_msgs_east(:,:,s), prod_of_msgs_west(:,:,s),...
    %				    pot, maximize);
    %msgs(:,:,:,:) = comp_msgs(old_msgs(:,:,:,:), prod_of_msgs(:,:,:), ...
    %				    old_msgs_east(:,:,:), old_msgs_west(:,:,:),...
    %				    prod_of_msgs_east(:,:), prod_of_msgs_west(:,:),...
    %				    pot, maximize);
    msgs(:,:,:,:) = comp_msgs(old_msgs, prod_of_msgs, ...
				    old_msgs_east, old_msgs_west,...
				    prod_of_msgs_east, prod_of_msgs_west,...
				    pot, maximize);

  %end
  
  % Compute local beliefs 
  %for s=1:nstrips
    % Take product of all incoming messages along all directions (dimension 3).
    % The result has size nrows*ncols*1*nstates*nstrips
    % Permute removes singleton dimension 3 (faster than squeeze).
    %prod_of_msgs(:,:,:,s)=permute(prod(msgs(:,:,:,:,s), 3), [1 2 4 3]);
    %prod_of_msgs(:,:,:,s)=prod_of_msgs(:,:,:,s) .* local_ev(:,:,:,s);
    prod_of_msgs(:,:,:)=permute(prod(msgs(:,:,:,:), 3), [1 2 4 3]);
    prod_of_msgs(:,:,:)=prod_of_msgs(:,:,:) .* local_ev(:,:,:);

    %old_bel(:,:,:,s) = new_bel(:,:,:,s);
    %new_bel(:,:,:,s) = normalize(prod_of_msgs(:,:,:,s), 3);
    %diff = (new_bel(:,:,:,s)-old_bel(:,:,:,s));
    %err(s) = max(diff(:));
    old_bel(:,:,:) = new_bel(:,:,:);
    new_bel(:,:,:) = normalize(prod_of_msgs(:,:,:), 3);
    diff = (new_bel(:,:,:)-old_bel(:,:,:));
    err = max(diff(:));
  %end
  
  %converged = max(err)<tol;
  %if verbose, fprintf('error at iter %d = %f\n', iter, sum(err)); end
  %iter = iter + 1;
end % for

%niter = iter-1;
niter = ITERATION;

%fprintf('converged in %d iterations\n', niter);

%bel = zeros(nrows, ncols, nstates);

%for s=1:nstrips
  %scols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip);
  %bel(:,scols,:) = new_bel(:,:,:,s);
%end

% Send data back to rank=0
if (my_rank == leader)
  bel = zeros(nrows, ncols, nstates);

  % Take care of local part.
  s = 1;
  scols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip);
  bel(:,scols,:) = new_bel(:,:,:);

  if (comm_size > 1)
    % Collect new beliefs.
    for i = 2:comm_size
      s = i;
      scols = ((s-1)*ncols_per_strip+1:s*ncols_per_strip);
      i_rank = i - 1;
      disp(['Receiving from: ' num2str(i_rank)]);
      bel(:,scols,:) = MPI_Recv(i_rank,output_tag_i,MPI_COMM_WORLD);
    end
  end
  MPI_Finalize;
end
if (my_rank ~= leader)
   MPI_Send(leader,output_tag_i,MPI_COMM_WORLD,new_bel);
   MPI_Finalize;
   exit;
end
