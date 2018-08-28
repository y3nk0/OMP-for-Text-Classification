close all; 
clear all; 
clc; 
options.Display = 0;%
addpath(genpath(pwd));

% datasets = {'data/sci.medsci.space','data/rec.sport.baseballrec.sport.hockey','data/alt.atheismsoc.religion.christian','data/comp.sys.ibm.pc.hardwarecomp.sys.mac.hardware','data/vote','data/movie','data/books','data/dvd','data/electronics','data/kitchen_&_housewares'};

datasets = {'data/sci.medsci.space'};

dataset_path = '';

for data=datasets
    
    dataset_name = strcat(dataset_path,data{1});
    
    corpus_path = strcat(dataset_name,'_unique_words_M_CORPUS.txt');
    labels_path = strcat(dataset_name,'_y_train.txt');

    X_train=load(strcat(dataset_name,'_train_bow_features.txt'));
    y_train=load(strcat(dataset_name,'_y_train.txt'));

    X_dev=load(strcat(dataset_name,'_dev_bow_features.txt'));
    y_dev=load(strcat(dataset_name,'_y_dev.txt'));

    X_test=load(strcat(dataset_name,'_test_bow_features.txt'));
    y_test=load(strcat(dataset_name,'_y_test.txt'));

    sparsity= @(w) (sum((w==0)));
    % counts # of zeroes

    % bias
    %X_train = [X_train, ones(size(X_train,1),1)]; % one more col to X
    %X_train = [ones(size(X_train,1),1), X_train]; % one more cDbtol to X
    y_train(y_train==0)=-1; % important!

    %X_dev = [X_dev, ones(size(X_dev,1),1)]; % one more col to X
    %X_dev = [ones(size(X_dev,1),1), X_dev]; % one more col to X
    y_dev(y_dev==0)=-1; % important!

    %X_test = [X_test, ones(size(X_test,1),1)]; % one more col to X
    %X_test = [ones(size(X_test,1),1), X_test]; % one more col to X
    y_test(y_test==0)=-1; % important!

    % M = [M, ones(size(M,1),1)]; % one more col to M

    pars = ['OMP'];
    %pars = ['l1'];
    
    norm_pars = cellstr(pars);

    lambdaVect = (100:100:2000);

    lambda2Vect = [10^-2 10^-1 1 10 100];
    %lambda2Vect = [1.00];
    
    rhoVect = lambdaVect;

    nVars = size(X_train,2);
    % funObj = @(w)LogisticLoss(w,X,y);

    for k=1:length(norm_pars)
        regularization=norm_pars{k};
        accs = zeros(length(lambda2Vect),1);
        percs = zeros(length(lambda2Vect),1);

        best_perc = 1.0;
        best_lambda = 0.0;
        best_acc = 0.0;
        counter = 1;

        if strcmp(regularization,'elastic')
            %lambda2Vect = lambdaVect;
            featureNetwork = 0;
            rhos = [0];
        else
            %lambda2Vect = [0];
            featureNetwork = 0;
            rhos = [0];
        end

        for rho=rhos
            for lambda2=lambda2Vect
                
                start = 0;
                active_indx_set = [];
                inactive_indx_set = [];
                Ar = [];

                win = [];
                A_T = [];
                wout = [];
                r = 0;
                normR = 0;
    
                for lambda=lambdaVect

%                     lambda = lambda*ones(nVars,1);
%                     lambda(1,1) = 0;
%                     lambda2 = 0;

%                     lambda2 = lambda*ones(nVars,1);
%                     lambda2(1,1) = 0;

                    if strcmp(regularization,'l1')
                        wout=logreg_regularized(X_train,y_train,lambda2);
                    elseif strcmp(regularization,'l2')                
                        wout=logreg_L2(X_train,y_train,lambda2);
                    elseif strcmp(regularization,'elastic')
                        wout = logreg_elastic(X_train,y_train,lambda,lambda2);
                    elseif strcmp(regularization,'OMP')
                        [wout,active_indx_set,inactive_indx_set,Ar,win,A_T,~,~] = OMP(X_train, y_train, start,active_indx_set,inactive_indx_set, lambda, lambda2,Ar,win,A_T,wout);
                        start = lambda;
                       
                    end

                    [~,acc] = predict(X_dev,y_dev,wout);
                    accs(counter,1) = acc;
                    perc = (size(wout,1)-sparsity(wout))/size(wout,1);
                    percs(counter,1) = perc;

                    if acc>best_acc
                        best_acc = acc;
                        best_lambda = lambda;
                        best_lambda2 = lambda2;
                        best_rho = rho;
                        best_wout = wout;
                        best_perc = perc;
                    elseif acc==best_acc
                        if perc < best_perc
                           best_lambda = lambda;
                           best_lambda2 = lambda2;
                           best_rho = rho;
                           best_perc = perc;
                           best_wout = wout;
                        end
                    end
                    fprintf('sparsity %f out of %f, perc: %f \n',sparsity(wout),size(wout,1),perc);
                    counter=counter+1;
               end
            end
        end

        if strcmp(regularization,'l1')
            [p,acc] = predict(X_test,y_test,best_wout);  
            acc_sparse_l1 = acc;
        elseif strcmp(regularization,'l2')
            [p,acc] = predict(X_test,y_test,best_wout);
            acc_l2 = acc;
        elseif strcmp(regularization,'elastic')
            [p,acc] = predict(X_test,y_test,best_wout);
            acc_elastic = acc;
        elseif strcmp(regularization,'OMP')
            [p,acc] = predict(X_test,y_test,best_wout);
            acc_omp = acc;
        end

        % l1_acc = acc;
        %save(strcat(data{1},'lasso_pred.txt'),'p','-ascii');
        p_las = importdata(strcat(data{1},'lasso_pred.txt'));
        
        perc = (size(best_wout,1)-sparsity(best_wout))/size(best_wout,1);
        
        %[B,I] = sort(best_wout,'descend');
        %B(1:10)
        %I(1:10)
        
        %save('omp_active.txt','I','-ascii');
        %save(strcat(data{1},'accs.txt'),'accs','-ascii');
        
        fprintf('sparsity %f out of %f, non-zero perc: %f \n',sparsity(best_wout),size(best_wout,1),perc);
    
        s = sum(p_las~=p);
        
        x=0;
        for d=1:size(y_test,1)
            if p(d,1)==y_test(d,1) && p_las(d,1)~=y_test(d,1)
                x = x+1;
            end
        end
       
        pout = myBinomTest(x,s,0.5,'Greater');
        if(pout<0.01)
            signif = 'True';
        else
            signif = 'False';
        end
        %signif = 'False';
        
    
    end

    fileID = fopen('parameters.txt','a');
    fprintf(fileID,'%s %s - Lambda1:%4.2f Lambda2:%4.2f rho:%4.2f acc:%4.4f perc:%4.4f signif:%s\n',dataset_name,regularization,best_lambda,best_lambda2,best_rho,acc,perc,signif);
    fclose(fileID);

end
