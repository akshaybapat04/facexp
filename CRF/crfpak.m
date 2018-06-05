function w = crfpak(net)
% crfpak Combine all the crf parameters into a ROW vector (as required by netlab)
% function w = crfpak(net)

bs = net.nparamsPerNodeEclass; % block size
wi = zeros(sum(bs),1);
for ec=1:net.nnodeEclasses
  wi(block(ec, bs)) = net.w{ec}(:);
end

adjustableEdges = find(net.adjustableEdgeEclassBitv);
bs = net.nstatesPerEdgeEclass(adjustableEdges); % block size
we = zeros(sum(bs),1);
for ec=adjustableEdges(:)'
  ec2 = find_equiv_posns(ec, adjustableEdges);
  pot = net.pot{ec}(:);
  we(block(ec2,bs)) = log(pot);
end

w = [wi; we];
w = w(:)'; 
