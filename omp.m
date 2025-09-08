function alpha = omp(D, x, sparsity)
    % OMP: Orthogonal Matching Pursuit for sparse coding
    
    residual = x;      % Initial residual
    idx_set = [];      % Set of indices of selected atoms
    alpha = zeros(size(D, 2), 1);  % Sparse coefficient vector
    
    for i = 1:sparsity
        % Project the residual onto the dictionary
        proj = D' * residual;
        
        % Find the index of the atom with the highest correlation
        [~, idx] = max(abs(proj));
        
        % Add the index to the set of selected indices
        idx_set = unique([idx_set, idx]);
        
        % Subset dictionary D to include only selected atoms
        D_sub = D(:, idx_set);
        
        % Compute the sparse coefficients for the selected atoms
        a = pinv(D_sub) * x;
        
        % Update the residual
        residual = x - D_sub * a;
        
        % If residual is sufficiently small, stop early
        if norm(residual) < 1e-6
            break;
        end
    end
    
    % Set the coefficients corresponding to the selected atoms
    alpha(idx_set) = a;
end
