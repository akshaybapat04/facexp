% We make an HMM with untied parameters
% and convert it to a CRF and check that inference gives the same results.
% This requires BNT.

% H1-H2-H3 
% |  |  |
% O1 O2 O3 
%

clear all

if 0
seed = 0;
rand('state',seed)
randn('state',seed)
end

% Make the HMM

T = 3; Q = 3; O = 2; cts_obs = 0; param_tying = 0;
bnet = mk_hmm_bnet(T, Q, O, cts_obs, param_tying); % BNT
ns = bnet.node_sizes;
hnodes = 1:T;
onodes = (1:T)+T;


% Convert to CRF

G = mk_undirected(bnet.dag);
G = G(hnodes, hnodes);
inputDims = repmat(O, 1, T);
nstates = repmat(Q, 1, T);
net = crf(inputDims, nstates, G);

CPD = struct(bnet.CPD{hnodes(1)});
pi = CPD.CPT;
CPD = struct(bnet.CPD{hnodes(2)});
A12 = CPD.CPT;
M = repmat(pi(:), 1, Q) .* A12;
e = net.E(hnodes(1), hnodes(2));
net.pot{e} = M;



for t=1:T
  CPD = struct(bnet.CPD{onodes(t)});
  B = CPD.CPT;
  net.w{t} = log(B' + (B'==0)*eps);
  if t>2
    CPD = struct(bnet.CPD{hnodes(t)});
    A = CPD.CPT;
    e = net.E(hnodes(t-1), hnodes(t));
    net.pot{e} = A;
  end
end


% Inference in the HMM (needs BNT)

data = cell2num(sample_bnet(bnet));
data = data(onodes);% data(i,s)
Nnodes= 2*T;
ev = cell(1,Nnodes);
ev(onodes) = num2cell(data);
engine = jtree_inf_engine(bnet);
[engine, loglik] = enter_evidence(engine, ev);

for t=1:T
  m = marginal_nodes(engine, hnodes(t));
  belH{t} = m.T;
  if t<T
    m = marginal_nodes(engine, [hnodes(t) hnodes(t+1)]);
    belHH{t} = m.T;
  end
end

% Inference in the CRF
X = encodeDiscrete(data', repmat(O,1,T)); % X{s,i}(:)

algos = {'bruteForceMrf2', 'bploopy', 'bptree'}; % bpchain assumes tied params
for a=1:length(algos)
  net = crfSetInfEngine(net, algos{a});
  [bel, belE, logZ] = crfinfer(net, X);
  
  assert(approxeq(loglik, logZ))
  for t=1:T
    assert(approxeq(belH{t}, bel{t}))
    if t<T
      e = net.E(hnodes(t), hnodes(t+1));
      assert(approxeq(belHH{t}, belE{e}))
    end
  end
end
