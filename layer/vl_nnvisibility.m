function [ y,dvdy ] = vl_nnvisibility( X,V,dzdy )
% VisibilityLayer
% X	: 112 x 112 x 3 x b
% V : 112 x 112 x 3 x b
% y : 112 x 112 x 3 x b

nbatch = size(X,4);

if nargin<3
    %forward
    y = zeros(size(X),'single');
    for i= 1:nbatch
        y(:,:,:,i) =  squeeze(V(:,:,:,i)) .* squeeze(X(:,:,:,i));
    end
    
else
    %backward
    y= zeros(size(X),'single');
    for i= 1:nbatch
        y(:,:,:,i) =  squeeze(V(:,:,:,i)) .* squeeze(dzdy(:,:,:,i));
    end
    
    dvdy= zeros(size(V),'single');
%     for i= 1:nbatch
%         dvdy(:,:,:,i) = squeeze(X(:,:,:,i)) .* squeeze(dzdy(:,:,:,i));
%     end
end

end

