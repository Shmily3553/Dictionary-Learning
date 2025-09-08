close all;clear all;
% obj=0.5*|X-FG|_F^2, st. F'F=I, |G|_0<=10
% Backtracking Line Search Method, for each step we increase the 
row=200;
column=100;
rank=50;
X=100*rand(row,column);
F=eye(row);
F=F(:,1:rank);
G=10*ones(rank,column);
F_old=F;G_old=G;
itr=500;
losses=zeros(itr,1);
alpha=0.5;
beta=0.618;
for i=1:itr
    grad_F=(F*G-X)*G';
    grad_G=F'*(F*G-X);
    L_F=norm(G*G');
    L_G=norm(F'*F);
    while true
        L_F=beta*L_F;
        F_temp=F-1/L_F*grad_F;
        [U,~,V]=svd(F_temp,0);
        F_temp=U*V';
        if (norm(X-F*G,'fro')^2-norm(X-F_temp*G,'fro')^2<alpha*norm(grad_F,'fro')^2)
            break
        end
    end
    L_F=L_F/beta;
    tmp_F=F-1/L_F*grad_F;
    [U,S,V]=svd(tmp_F,0);
    F=U*V';
    while true
        L_G=beta*L_G;
        tmp_G=G-1/L_G*grad_G;
        G_temp=myMaxk(tmp_G,10);
        if (norm(X-F*G,'fro')^2-norm(X-F*G_temp,'fro')^2<alpha*norm(grad_G,'fro')^2)
            break
        end
    end
    L_G=L_G/beta;    
    tmp_G=G-1/L_G*grad_G;
    G=myMaxk(tmp_G,10);
    losses(i)=.5*norm(X-F*G,'fro')^2;
end

F=F_old;G=G_old;
losses1=zeros(itr,1);
for i=1:itr
    grad_F=(F*G-X)*G';
    grad_G=F'*(F*G-X);
    L_F=norm(G*G');
    L_G=norm(F'*F);
    tmp_F=F-1/L_F*grad_F;
    [U,S,V]=svd(tmp_F,0);
    F=U*V';
    tmp_G=G-1/L_G*grad_G;
    G=myMaxk(tmp_G,10);
    losses1(i)=.5*norm(X-F*G,'fro')^2;
end
plot(losses,'k','LineWidth',2.5,'MarkerSize',20);
hold on;
plot(losses1,'r','LineWidth',2.5,'MarkerSize',20);
grid on