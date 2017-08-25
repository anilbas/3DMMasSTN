function [ y,drdy ] = vl_nnrotate3D( X,R, dzdy )
% Rotate3DPointsLayer
% X	: 1 x 3 x n x b
% R : 1 x 3 x 3 x b
% y : 1 x 3 x n x b

nbatch = size(X,4);

if nargin<3
    %forward
    y = zeros(size(X),'single');
    for i= 1:nbatch
        y(1,:,:,i) =  squeeze(R(1,:,:,i)) * squeeze(X(1,:,:,i));
    end
else
    %backward
    y= zeros(size(X),'single');
    for i= 1:nbatch
        y(1,:,:,i) =  squeeze(R(1,:,:,i))' * squeeze(dzdy(1,:,:,i));
    end
    
    drdy= zeros(size(R),'single');
    for i= 1:nbatch
        drdy(1,:,:,i) = ( squeeze(X(1,:,:,i)) * squeeze(dzdy(1,:,:,i))' )';
    end
end

end

