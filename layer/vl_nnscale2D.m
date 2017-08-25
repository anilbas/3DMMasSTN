function [ y,dsdy ] = vl_nnscale2D(  X, S, dzdy )
% Scale2DPointsLayer
% X	: 1 x 2 x n x b
% S : 1 x 1 x 1 x b
% y : 1 x 2 x n x b

if nargin<3
    %forward
    y = bsxfun(@times,X,S);
else
    %backward
    dx = repmat(S, [1 size(X,2) size(X,3) 1]);
    
    y = dzdy.*dx;
    dsdy = sum(sum(dzdy.*X,2),3);
end

end

