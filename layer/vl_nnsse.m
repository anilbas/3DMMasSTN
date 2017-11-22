function [ y ] = vl_nnsse( x,p )
% sumofsquarederrors Loss Layer
if nargin<2
    y = sum(sum(x.^2));
else
    dx = 2 * x;
    y = p*dx;
end

end


