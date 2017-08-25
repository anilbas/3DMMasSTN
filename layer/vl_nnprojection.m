function [ y ] = vl_nnprojection( X, dzdy )
% ProjectionLayer
% X	: 1 x 3 x n x b
% y : 1 x 2 x n x b

if nargin<2
    %forward
    y = zeros(1,2,size(X,3),size(X,4),'single');
    y(1,1:2,:,:) = X(1,1:2,:,:);
else
    %backward
    y = zeros(size(X),'single');
    y(1,1:2,:,:) = dzdy; 
end

end

