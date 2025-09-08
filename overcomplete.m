% function Dictionary = overcomplete(X, U0, V0, itr, topK)
% obj=0.5*|X-FG|_F^2, st. F'F=I, |G|_0<=10
% Backtracking Line Search Method, for each step we increase the 

% change from orthogonal dictionary learning to overcomplete one
%   argmin 0.5*||X-FG||_F^2 
%    s.t. ||F_i||_2 <= 1, ||G_j||_0<=s
%     F,G

% backtracking (haven't changed)
% F = U0;G = V0;
row=200;
column=100;
rank=50;
X=100*rand(row,column);
F=eye(row);
F=F(:,1:rank);
G=10*ones(rank,column);
F_old=F;G_old=G;
topK = 10;
itr=50;
% losses=zeros(itr,1);
alpha=0.5;
beta=0.618;
for i=1:itr
    grad_F=(F*G-X)*G';
    grad_G=F'*(F*G-X);
    L_F=norm(G*G');
    L_G=norm(F'*F);
    % update F (ortho)
    while true
        L_F=beta*L_F;
        F_temp=F-1/L_F*grad_F;
        [U,~,V]=svd(F_temp,0);
        F_temp=U*V';
        if (norm(X-F*G,'fro')^2-norm(X-F_temp*G,'fro')^2<alpha/L_F*norm(grad_F,'fro')^2)
            break
        end
    end
    L_F=L_F/beta;
    tmp_F=F-1/L_F*grad_F;
    [U,~,V]=svd(tmp_F,0);
    F=U*V';
    % update G (sparsity)
    while true
        L_G=beta*L_G;
        tmp_G=G-1/L_G*grad_G;
        G_temp=myMaxk(tmp_G,topK);
        if (norm(X-F*G,'fro')^2-norm(X-F*G_temp,'fro')^2<alpha/L_G*norm(grad_G,'fro')^2)
            break
        end
    end
    L_G=L_G/beta;    
    tmp_G=G-1/L_G*grad_G;
    G=myMaxk(tmp_G,topK);
    % losses(i)=.5*norm(X-F*G,'fro')^2;
end

% vanilla
F=F_old;G=G_old;
losses1=zeros(itr,1);
for i=1:itr
    grad_F=(F*G-X)*G';
    grad_G=F'*(F*G-X);
    L_F=norm(G*G');
    L_G=norm(F'*F);
    % update F
    tmp_F=F-1/L_F*grad_F;
    % [U,~,V]=svd(tmp_F,0);
    % F=U*V';
    col_norm = vecnorm(tmp_F);
    F = tmp_F ./ max(col_norm, 1);
    % update G
    tmp_G=G-1/L_G*grad_G;
    G=myMaxk(tmp_G, topK);
    losses1(i)=.5*norm(X-F*G,'fro')^2;
end
% plot(losses,'k','LineWidth',2.5,'MarkerSize',20);
% hold on;
figure;
semilogy(1:itr,losses1-losses1(end),'r','LineWidth',2.5,'MarkerSize',20);
grid on
% legend('Backtracking','Vanilla')