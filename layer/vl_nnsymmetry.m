function [ y ] = vl_nnsymmetry( x,p )
%Symmetry Loss Layer
%figure; subplot(1,2,1); imshow(double(x(:,:,:,1))./255);
%subplot(1,2,2); imshow(double(fliplr(x(:,:,:,1)))./255);
if nargin<2
    delta = (x - fliplr(x) ).* (x~=0 & fliplr(x)~=0);
    y = sum(delta(:).^2) / sum(x(:)~=0) ;
else
    y = 2 * p * ( (x - fliplr(x) ).* (x~=0 & fliplr(x)~=0) ) ;
end

end