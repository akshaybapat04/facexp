function w = crfchainpak(net)
% crfpak Combine all the crf parameters into a ROW vector (as required by netlab)
% function w = crfpak(net)

w = [net.w(:); log(net.pot(:))]';
