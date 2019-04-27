function [w] = logreg_regularized(X,y,lambda_las)
tic;

[nInst,nVars]= size(X);
w_init = zeros(nVars,1);

lambda = lambda_las*ones(nVars,1);
lambda(1,1) = 0;

% lambda = initVal*ones(nVars,1);
funObj = @(w)LogisticLoss(w,X,y);
% funObj = @(t)l1Obj(t, X, y, lambda);
fprintf('\nComputing L1-Regularized Logistic Regression Coefficients...\n');
w=L1General2_PSSgb(funObj,w_init,lambda);
time = toc;