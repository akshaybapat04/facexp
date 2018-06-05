function [localEvHMM] = lattice2Ev_to_hmmEv(localEvMRF2)
% Given localEvMRF2(r,c,:), compute localEvHMM(:,c) by combining all evidences for each row
% Example:
% 1 - 4
% |   |
% 2 - 5
% |   |
% 3 - 6
% 
% localEvHMM([s1 s2 s3],t) = localEvMRF1(1,t,s1) * localEvMRF1(2,t,s2) * localEvMRF1(3,t,s3)

[nr nc K] = size(localEvMRF2);
localEvHMM = ones(K^nr, nc);
bigdom = 1:nr;
bigsz = K*ones(1,nr);
for c=1:nc 
  tmp = myones(bigsz);
  for r=1:nr
    smalldom = r;
    smallsz = K;
    tmp = mult_by_table(tmp, bigdom, bigsz, localEvMRF2(r,c,:), smalldom, smallsz);
  end
  
  if 0
  % check
  for s=1:K^nr
    p = 1;
    vals = ind2subv(bigsz, s);
    for r=1:nr
      p = p*localEvMRF2(r,c,vals(r));
    end
    tmp2(s) = p;
  end
  assert(approxeq(tmp, tmp2))
  end
  
  localEvHMM(:,c) = tmp(:);
end
