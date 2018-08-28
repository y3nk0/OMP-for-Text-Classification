function [obj,g]=l1Obj(w,X,y,lambda)
[obj,g] = LogisticLoss(w,X,y); 
[m,n] = size(X);

f = @(t)(sigmoid(t));

% obj = ((1/m)*(-y'*log(f(X*w))-(1-y)'*log(1-f(X*w))));

tmp = sum((lambda(2)/(2.0*m)).*(abs(w(2:end))));
obj = obj + tmp;

grad(1,:) = (1/m) * (f(X*w)-y)'*X(:,1);
j=2:size(w);
grad(j,:) = (1/m) * ((f(X*w)-y)'*X(:,j)) + ((lambda(2)/m) .* w(j))';

% g = g + rho.*N.*(w-MyMu);
g = grad;