function [obj, g]=lassoObj(w,v,X,y,M,rho,u,N,MyMu,lambda)
[obj,g] = LogisticLoss(w,X,y); 

tmp = (rho/2)*norm(M*w-(v+ (u./rho))).^2+sum(lambda.*abs(w));
obj = obj + tmp; 
g = g + rho.*N.*(w-MyMu);
