function [A]=gridNeighbors(gsize)

ssize=[gsize gsize];

N=gsize*gsize;


A=sparse(N,N);


for i=1:gsize
  for j=1:gsize
    p1=sub2ind(ssize,i,j);
    if (i>1)
      pup=sub2ind(ssize,i-1,j);
      A(p1,pup)=1;
    end
    if (i<gsize)
      pdown=sub2ind(ssize,i+1,j);
      A(p1,pdown)=1;
    end
    if (j<gsize)
      pleft=sub2ind(ssize,i,j+1);
      A(p1,pleft)=1;
    end
    if (j>1)
      pright=sub2ind(ssize,i,j-1);
      A(p1,pright)=1;
    end
  end
  
end
A=(A+A')>0;
    
    
 