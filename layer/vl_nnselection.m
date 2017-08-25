function [ y ] = vl_nnselection( X,idx,dzdy )
% SelectionLayer
% X	: 1 x 2 x n x b
% y : 1 x 2 x nidx x b

if nargin<3
    %forward
    y = zeros(1,size(X,2),length(idx),size(X,4),'single');
    y(1,:,:,:) = X(1,:,idx,:);
else
    %backward
    y = zeros(size(X),'single');
    y(1,:,idx,:) = dzdy;
end

end

