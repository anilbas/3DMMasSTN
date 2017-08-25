function [ y ] = vl_nnsiamese( x,p )
%Siamese Loss Layer
%figure; subplot(1,2,1); imshow(double(x(:,:,:,1))./255);
%subplot(1,2,2); imshow(double(x(:,:,:,2))./255);
if nargin<2
    delta = (x(:,:,:,1:2:end) - x(:,:,:,2:2:end) ).* ((x(:,:,:,1:2:end)~=0) & (x(:,:,:,2:2:end)~=0));
    y = sum(delta(:).^2) / sum(x(:)~=0) ;
else
    y = 2 * p * ( (x(:,:,:,1:2:end) - x(:,:,:,2:2:end)) .* ((x(:,:,:,1:2:end)~=0) & (x(:,:,:,2:2:end)~=0)) );
    y =repelem(y,1,1,1,2);
end

end


