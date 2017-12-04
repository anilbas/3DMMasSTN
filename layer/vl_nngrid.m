function [ y ] = vl_nngrid( X,dzdy )
% GridLayer
% X	: 1 x 2 x N x b
% y : 2 x Ho x Wo x b
nbatch = size(X,4);
grid_dim = sqrt(size(X, 3)); % Usually this will be 224
if nargin<2
    %forward
    
    % We're centering the grid here and rescaling to a [-1 1] coordinate
    % system. The input image is 224x224x3.
    Xo = (X - 112)/112; 
    
    Xo = reshape(Xo,2,grid_dim,grid_dim,nbatch);
    newgrid(1,:,:,:) = -Xo(2,:,:,:);
    newgrid(2,:,:,:) =  Xo(1,:,:,:);
    y = permute(newgrid,[1,3,2,4]);

else
    %backward
    dY = permute(dzdy, [1,3,2,4]);
    Yo(1,:,:,:) = dY(2,:,:,:);
    Yo(2,:,:,:) =  -dY(1,:,:,:);
    Yo = reshape(Yo,1,2,grid_dim^2,nbatch);
    y = (1/112).*Yo;
end

end

