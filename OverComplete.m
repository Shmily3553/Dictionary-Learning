% change from orthogonal dictionary learning to overcomplete one
%   argmin 0.5*||X-FG||_F^2 
%    s.t. ||F_i||_2 <= 1, ||G_j||_0<=s
%     F,G

function F = OverComplete(X, F, G, itr, topK)
    % vanilla
    for i=1:itr
        grad_F = (F*G-X)*G';
        grad_G = F'*(F*G-X);
        L_F = normest(G*G');
        L_G = norm(F'*F);
        % update F
        tmp_F = F-1/L_F*grad_F;
        col_norm = vecnorm(tmp_F);
        F = tmp_F ./ max(col_norm, 1);
        % update G
        tmp_G = G-1/L_G*grad_G;
        G = myMaxk(tmp_G, topK);
    end    
end