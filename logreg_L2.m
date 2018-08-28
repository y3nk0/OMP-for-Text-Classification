function [wout,g]=logreg_L2(X,y,lambda, w_init)
% log reg with L2 regularization
%tic;
[~,nVars]= size(X);
funObj = @(w)LogisticLoss(w,X,y);
%funObj = @(w)costFunctionN(w, X, y);
funObjL2 = @(w)penalizedL2(w,funObj,lambda);
if nargin < 4
    w_init = zeros(nVars,1);
end
%fprintf('\nRunning Scaled Conjugate Gradient\n');
options.Method = 'scg';
options.Display = 'off';
%options = optimset('Display', 'off');
 
%[wout,g] = minFunc(@ridgeObj,w_init,options,X,y,lambda);
%wout = minFunc(@penalizedL2,zeros(nVars,1),options,funObj,lambda);
wout = minFunc(funObjL2,w_init,options);
%time = toc;
g=0;