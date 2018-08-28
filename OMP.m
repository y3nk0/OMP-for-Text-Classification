function [w,active_indx_set,inactive_indx_set,Ar,win,A_T,residHist, errHist] = OMP( A, b, start,active_indx_set,inactive_indx_set, K, lambda,Ar,win,A_T,w,opts )

X = A;
for i = 1:size(X,2)
    A(:,i) = orth(X(:,i));
end
A(:,1) = 1;

y = b;
y(b == -1) = 0;

if nargin < 13, opts = []; end
if ~isempty(opts) && ~isstruct(opts)
    error('"opts" must be a structure');
end

function out = setOpts( field, default )
    if ~isfield( opts, field )
        opts.(field) = default;
    end
    out = opts.(field);
end

printEvery  = setOpts('printEvery', 100);

% What stopping criteria to use? either a fixed # of iterations,
%   or a desired size of residual:
% target_resid    = -Inf;
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
r           = y;            % Initial residual
normR       = norm(r);      % Norm of residual

% N           = size(Ar,1);   % Number of features
% M           = size(r,1);    % Number or samples, observations
N = size(A,2);
M = size(A,1);

if K > N
    error('K cannot be larger than the dimension of the atoms');
end

if start==0
    w = zeros(N,1);
    Ar = A'*r;         % Correlation
    active_indx_set           = [];  
    inactive_indx_set         = 1:N;
    win = [];
    A_T = zeros(M,K);
    
end

naa = start; % Number of active atoms

residHist           = zeros(K,1);
errHist             = zeros(K,1);

fprintf('Iter,  Resid\n'); 

tic;
nk = 1;
while(1)    
    % -- Step 1: find new index and atom to add
    for in_naa = 1:nk
        if any(1==inactive_indx_set)
            indx = 1;
        else
            [mvalue,indx] = max(abs(Ar));
        end
        
        naa = naa + 1;
        
        new_atom                        = inactive_indx_set(indx);
        inactive_indx_set(indx)         = [];
        Ar(indx)                        = [];
        
        active_indx_set(naa)            = new_atom;
        
        A_T(:,naa) = A(:,new_atom);
        
        if(naa >= K)
            break
        end
       
    end
    
    AA = A_T(:,1:naa);
    
    % -- Step 2: Update Residual    
    %LL = .1*ones(size(AA,2),1);
    %win = logreg_regularized(AA,b,LL);
    
    
    win = logreg_L2(AA, b, lambda, [win; zeros(in_naa,1)]); 
    
    Xw = AA*win;
    
    r = ((1./(1.+exp(-Xw))) - y);
    normR = norm(r);
  
    % -- Print some info --
    PRINT   = (~mod( naa, printEvery ) || naa == K);
    %if printEvery > 0 && printEvery < Inf % && (normR < target_resid )
    %    % this is our final iteration, so display info
    %    PRINT = true;
    %end
    
    if PRINT, fprintf('%4d, %.2e\n', naa, normR ); end
    residHist(naa)   = normR;
    
    %if normR < target_resid
    %    if PRINT
    %        fprintf('Residual reached desired size (%.2e < %.2e)\n', normR, target_resid );
    %    end
    %    break;
    %end
    
    Ar  = (A(:,inactive_indx_set)'*r).^1.;
    
    if naa >= K
        break;
    end
%     else
%         Ar  = (A(:,inactive_indx_set)'*r).^1.; % prepare for next round
%     end
end
toc
%if (target_resid) && ( normR >= target_resid )
%    fprintf('Warning: did not reach target size of residual\n');
%end

w(active_indx_set) = logreg_L2(X(:,active_indx_set),b, lambda);

end % end of main function
