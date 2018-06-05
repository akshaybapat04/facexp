

K = 2;
nr = 3;
nc = 4;
local_evidence = rand(nr, nc, K);
%local_evidence = ones(nr, nc, K);

kernel = rand(K,K);
kernel = kernel+kernel'; % make symmetric



[bel, bel2, negloglik] = brute_force_inf_lattice2(kernel, local_evidence);

[belHMM, bel2HMM, negloglikHMM] = lattice2_hmm_inf(kernel, local_evidence);


approxeq(belHMM, bel)
approxeq(bel2HMM, bel2)
approxeq(negloglikHMM, negloglik)

