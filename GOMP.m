function [w,active_indx_set,r,win,residHist, errHist] = GOMP(A, b, K, lambda, groups, start,active_indx_set,r,w,win)

X = A;
la = length(groups);
I = ones(la);
parfor i = 1:la
    groups{i} = cell2mat(cellfun(@str2num, groups{i}, 'un', 0));
end

parfor i = 1:size(X,2)
    groups{la + i} = i;
    A(:,i) = orth(X(:,i));
end

A(:,1) = 1;
features = 1:size(A,2);
y = b;
y(b == -1) = 0;

% 
% if nargin < 5, opts = []; end
% if ~isempty(opts) && ~isstruct(opts)
%     error('"opts" must be a structure');
% end
% 
% function out = setOpts( field, default )
%     if ~isfield( opts, field )
%         opts.(field) = default;
%     end
%     out = opts.(field);
% end
% printEvery  = setOpts('printEvery', 100);
printEvery = 100;
% What stopping criteria to use? either a fixed # of iterations,
%   or a desired size of residual:
target_resid    = -Inf;
% if iscell(K)
%     target_resid = K{1};
%     K   = size(b,1);
% elseif K ~= round(K)
%     target_resid = K;
%     K   = size(b,1);
% end

% (the residual is always guaranteed to decrease)
target_resid = 1e-03;
if target_resid == 0 
    if printEvery > 0 && printEvery < Inf
        disp('Warning: target_resid set to 0. This is difficult numerically: changing to 1e-12 instead');
    end
    target_resid    = 1e-12;
end

% -- Intitialize --
% start at x = 0, so r = b - A*x = b

N = size(A,2);

if K > N
    error('K cannot be larger than the dimension of the atoms');
end

if start==0
    w = zeros(N,1);
    active_indx_set           = [];  
    inactive_indx_set         = 1:N;
    r           = b;            % Initial residual
    normR       = norm(r);      % Norm of residual
    win = [];
end

residHist           = zeros(K,1);
errHist             = zeros(K,1);

fprintf('Iter,  Resid\n'); 

tic;

lnew_atoms = start;

    
while(1)
    lg = length(groups);
    gv = zeros(1,lg);
    %%-- Step 1: find new index and atom to add
    parfor g = 1:lg
        groups{g} = setdiff(groups{g}, active_indx_set); 
        AG = A(:,groups{g});
        la_AG = size(AG,2);
        %gv(g) = mean(abs(AG'*r));
        gv(g) = (r'*AG*(pinv(AG'*AG + eye(la_AG)*0.01))*AG'*r)/la_AG;
        %gv(g) = r'*AG*(pinv(AG'*AG))*AG'*r;
    end
    
    [mvalue, gindx] = max(gv);
    new_atoms = setdiff(groups{gindx}, active_indx_set);
    lnew_atoms = lnew_atoms + length(new_atoms);
    active_indx_set = [active_indx_set, new_atoms];
    %inactive_indx_set = setdiff(features, active_indx_set); 
        
    groups{gindx} = [];
    groups(all(cellfun('isempty',groups),2),:) = [];
    
    % -- Step 2: Update Residual    
    AA = A(:,active_indx_set);
    win  = logreg_L2(AA, b, lambda); 
    %win  = logreg_L2(AA, b, lambda, [win; zeros(length(new_atoms),1)]); 
    Xw   = AA*win; 
    
    r = ((1./(1.+exp(-Xw))) - y);
    %r = (exp(Xw)./(1.+exp(Xw))) - y;
    normR = norm(r);
    
    % -- Print some info --
    PRINT   = ( ~mod( lnew_atoms, printEvery ) || lnew_atoms == K );
    %if printEvery > 0 && printEvery < Inf % && (normR < target_resid )
    %    % this is our final iteration, so display info
    %    PRINT = true;
    %end
    
    if PRINT, fprintf('%4d, %.2e\n', lnew_atoms, normR ); end
    residHist(length(active_indx_set))   = normR;
    
    if length(active_indx_set) >= K
        length(active_indx_set)
        break;
    end
    
end
toc
%if (target_resid) && ( normR >= target_resid )
%    fprintf('Warning: did not reach target size of residual\n');
%end

w(active_indx_set) = logreg_L2(X(:,active_indx_set),b, lambda);

end
