function [wout,g]=logreg_L1(X,y,lambda_las)
% log reg with L1 regularization
[~,nVars]= size(X);

w_init = zeros(nVars,1);

lambda = lambda_las*ones(nVars,1);
lambda(1,1) = 0;

fprintf('\nRunning Scaled Conjugate Gradient\n');
options.Method = 'scg';
options.MaxIter = 10000;

% options = optimset('GradObj', 'on', 'MaxIter', 1000);
% g=0;
% [wout,g] = fminunc(@(t)(l1Obj(t, X, y, lambda)), w_init, options);

[wout,g] = minFunc(@l1Obj,w_init,options,X,y,lambda);
