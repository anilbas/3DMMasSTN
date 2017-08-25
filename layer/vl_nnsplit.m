function [ y,r,t,s ] = vl_nnsplit( X,dx,dr,dt,ds )
% SplitLayer
% X	: 1 x 1 x n x b
% y : 1 x 1 x 10 x b

if nargin<2
    %forward
    r(1,1,1:3,:) = X(1,1,1:3,:);
    t(1,1:2,1,:) = X(1,1,4:5,:);
    s(1,1,1,:)   = X(1,1,6,:);
    y(1,1,1:10,:)= X(1,1,7:16,:);
else
    %backward
    y = zeros(size(X),'single');
    for i=1:size(X,4)        
        temp = cat(3,dr(:,:,:,i), reshape(dt(:,:,:,i),1,1,2), ds(:,:,:,i), dx(:,:,:,i));
        y(1,1,:,i) = temp;
    end
end

end

