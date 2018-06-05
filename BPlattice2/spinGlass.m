function [adjMatrix,Psi,Ls,J]=spinGlass(isize,LsSig,PsiSig)

% make a spin glass of size isize
% adjMatrix(i,j)
% Psi{e}(q,q')
% Ls{i}(q)


adjMatrix=gridNeighbors(isize);
[E, Nedges] = assignEdgeNums(adjMatrix);

N=isize^2;
for i=1:N
   jj=randn(1)*LsSig;
   Ls{i}=[exp(-jj);exp(jj)];
end

for i=1:N
   for j=i+1:N % 1:N
      if (adjMatrix(i,j)>0)
         jj=randn(1)*PsiSig;
	 J(i,j)=jj;
         M=[exp(-jj) exp(jj);
            exp(jj) exp(-jj)];
         %Psi{i,j}=M;
         %Psi{j,i}=M;
	 Psi{E(i,j)} = M;
      end
   end
end

