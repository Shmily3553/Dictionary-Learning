function [ C ] = myMaxk( A,b )
%MYMAXK Summary of this function goes here
%   Detailed explanation goes here
    signs=sign(A);
    AA=abs(A);
    [result,idx]=sort(AA,1);
    idx_mask=idx(1+size(idx,1)-b:end,:);
    mask_1=zeros(size(A));
    for j=1:size(idx_mask,2)
        for i=1:size(idx_mask,1)
            mask_1(idx_mask(i,j),j)=1;
        end
    end
    C=AA.*signs.*mask_1;