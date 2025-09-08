% l2dl:  L2 norm based dictionary learning with acceleration
% D= l1dl(X, theta, U0, V0, n_iter, topK, mu, lambda) sovles
%   argmin ||X-UV||_F^2 + theta ||V||_F^2 
%    s.t. ||V_j||_0<=s
%     U,V
%   - Input
%       - X: input data matrix with each column as an observation
%       - theta: regularization parameter on code sparsity
%       - U: initial guess of dictionary
%       - V: initial guess of sparse code
%       - n_iter: number of iterations
%       - topK: s (sparsity constraint)
%       - mu: parameter in dictionary proximal term
%       - lambda: parameter in coefficient matrix proximal term
%   - Output
%       - D: learned dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use of this code is free for research purposes only.


function Dictionary = Ours(X, theta, U0, V0, n_iter, topK, mu, lambda)
Dictionary=U0;
CoefMatrix=V0;

[rownumU,colnumU]=size(U0);

for k = 1:n_iter
    %update Dictionary
    M = 2*(X-Dictionary*CoefMatrix)*transpose(CoefMatrix) + Dictionary.*mu;
    %ensure rownum>colnum
    if rownumU < colnumU
        [P,S,Q]=svd(M',0);
        Dictionary = transpose(P*Q');
    else
        [P,S,Q]=svd(M,0);
        Dictionary = P*Q';
    end
    %update CoefMatrix
    CoefMatrix= ((lambda-theta-1).*CoefMatrix+Dictionary'*X)./lambda;
    CoefMatrix= myMaxk(CoefMatrix,topK);
   % l2error= norm(X-Dictionary*CoefMatrix, 'fro')+delta*norm(CoefMatrix, 'fro')
end