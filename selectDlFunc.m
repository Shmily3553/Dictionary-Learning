%   This program selects the corresponding algorithm to solve the object function 
%   - Input
%       - X: input data matrix with each column as an observation
%       - lambda: regularization parameter on code sparsity
%       - U: initial guess of dictionary 
%       - V: initial guess of sparse code
%       - UA: size of A in FODL
%       - UD: initial guess of FODL dictionary
%       - opts: options
%           - n_iter: number of iterations
%       - std_noise: standard deviation of noise
%       - fcnhandle: the function to call  available choices: (FODL, LNDL, Ours)
%       - topK: the top s highest absolute values mentioned in the paper
%   - Output
%       - Dictionary: learned dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use of this code is free for research purposes only.


function Dictionary = selectDlFunc(X, lambda, U0, V0, opts, opts_ksvd ,opts_mod, UA, std_noise, UD, fcnhandle, topK, mu, lambda2, n_atom)
    func = str2func(strcat('@',fcnhandle));
    if fcnhandle=="FODL"
       Dictionary=func(X, std_noise, UD, V0, UA, opts.n_iter);
    elseif fcnhandle=="LNDL"
        Dictionary=l0dl(X, lambda, U0, V0, opts);
    elseif fcnhandle=="overcomplete"
        Dictionary=OverComplete(X, U0, V0, opts.n_iter, topK);
    elseif fcnhandle=="mod"
        [Dictionary,~,~]=perform_dictionary_learning(X, opts_mod);
    elseif fcnhandle=="ksvd"
        [Dictionary,~,~]=perform_dictionary_learning(X, opts_ksvd);
    else 
        Dictionary=func(X, lambda,  U0, V0, opts.n_iter, topK, mu, lambda2);
    end