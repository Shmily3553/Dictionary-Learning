%   This program select the algorithm to denoise the picture 
%   - Input
%       - D: learned dictionary
%       - A: noisy image patches
%   - Output
%       - im_denoised: the denoised image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Use of this code is free for research purposes only.


function im_denoised= selectDenoisedFunc(D, A, sz_patch, sz_gt,epsilon, lambda1,fcnhandle)
    %func = str2func(fcnhandle);
    if fcnhandle=="FODL"
        CoefMatrix=D'*A;
        for i = 1:size(CoefMatrix,2)
            AA=CoefMatrix(:,i);
            B=wthresh(AA,"h",lambda1);
            CoefMatrix(:,i)=B;
        end
        G= D*CoefMatrix;
        im_denoised = cols2im(G, sz_gt);
    else 
        Y = A;
        M = repmat(mean(Y,1),[size(Y,1) 1]);
        Y = Y - M;
        C = omp2(D'*Y,sum(Y.^2,1),D'*D,epsilon,'maxatoms',floor(sz_patch^2/2),'checkdict','off'); 
        im_denoised = cols2im(D*C+M, sz_gt);
    end