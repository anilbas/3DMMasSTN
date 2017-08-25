function [ y ] = vl_nneuclideanloss( x,r,vis,p )
% Euclidean Loss Layer
if nargin<4
    %forward
    %delta = x - r ;
    %y = sum(delta(:).^2) ;
    
    %vis=ones(1,1,21,1,'single');
    
    delta = (x - r).^2 ;
    temp(1,1,:,:)=vis;
    temp(1,2,:,:)=vis;
    delta = delta.*temp;
    y = sum(delta(:));
    
else
    %backward
    %y = 2 * p * (x - r) ;
    
    %vis=ones(1,1,21,1,'single');
    dx = 2 * (x - r) ;
    dx(:,1,:,:)=dx(:,1,:,:).*vis;
    dx(:,2,:,:)=dx(:,2,:,:).*vis;
    y = p*dx;
    
end

end


