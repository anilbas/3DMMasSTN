function [ y,dtdy ] = vl_nntranslate2D( X,T, dzdy )
% Translate2DPointsLayer
% X	: 1 x 2 x n x b
% T : 1 x 2 x 1 x b
% y : 1 x 2 x n x b

if nargin<3
    %forward
    y = bsxfun(@plus, X, T);
else
    %backward
    y = dzdy;
    dtdy = sum(dzdy,3);
    
end

end

