function [p,acc]=predict(X,y,w)
Xw = X*w;
sig = 1./(1+exp(-Xw));
sig(sig>=0.5)=1;
if sum(y==-1)>0
    sig(sig<0.5)=-1;
    p=sig;
    errors = p+y;
    error = sum(errors==0);
else
    sig(sig<0.5)=0;
    p=sig;
    error=sum(abs(p-y));
end

total=size(y,1);
acc=(total-error)/total;
fprintf('error %f out of %f ,acc: %f \n',error,total,acc);


