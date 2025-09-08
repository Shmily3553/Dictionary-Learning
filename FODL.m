% online:  Fast sparsity-based orthogonal dictionary learning for image restoration
% D= l1dl(X, std_noise, U0, V0, UA, n_iter) sovles
%   argmin ||X-UV||_F^2 + lamda*||V||_0
%     U,V
%   - Input
%       - X: input data matrix with each column as an observation
%       - delta: regularization parameter on code sparsity
%       - U: initial guess of dictionary [A;D]
%       - V: initial guess of sparse code
%       - UA: size of A
%       - opts: options
%           - n_iter: number of iterations
%   - Output
%       - D: learned dictionary
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reference: Chenglong Bao, Jian-Feng Cai and Hui Ji
%Fast sparsity-based orthogonal dictionary learning for image restoration
%ICCV2013
%-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use of this code is free for research purposes only.
%
%Author:  
%
%Last Revision: 20-Nov-2019

function Dictionary = FODL(X, std_noise, U0, V0, UA, n_iter)
Dictionary=U0;
lambda=3.5*std_noise;
[rownumU, colnumU] = size(Dictionary);
CoefMatrix=zeros(colnumU,1);
[rownumV, colnumV] = size(V0);

for k = 1:n_iter
    %update CoefMatrix
    Temp=Dictionary'*X;
    for i = 1:colnumV
        A=Temp(:,i);
        for j = 1:rownumU
            if (A(j,1)>lambda)
                CoefMatrix(j,i)=A(j,1);
            else
                CoefMatrix(j,i)=0;
            end
        end
    end
    %CoefMatrix = sparse(wthresh(Dictionary'*X, 'h', lamda));
  
    %update Dictionary
    M = X*CoefMatrix(UA+1:end,:)'-Dictionary(:,UA)*(Dictionary(:,UA)'*X*CoefMatrix(UA+1:end,:)');

    [rownumM,colnumM]=size(M);
    if rownumM <= colnumM
        [P,S,Q]=svd(M',0);
        Dictionary(:,UA+1:end) = transpose(P*Q');
    else
        [P,S,Q]=svd(M,0);
        Dictionary(:,UA+1:end) = P*Q';
    end
    %ICCVerror= norm(X-Dictionary*CoefMatrix, 'fro')+lambda*lambda*nnz(CoefMatrix)   
end
