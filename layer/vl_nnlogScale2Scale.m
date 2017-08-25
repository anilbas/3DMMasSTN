function [ y ] = vl_nnlogScale2Scale( s, dzdy )
% logScale2ScaleLayer
% X	: 1 x 1 x 1 x b
% y : 1 x 1 x 1 x b

if nargin<2
    %forward
    y=(99/100).^s;
    %y = exp(s);
    %y = s.^2;
    %y = abs(s);
else
    %backward
    y = dzdy.*(99/100).^s*log(99/100);
    %y = dzdy.*exp(s);
    %y = dzdy.*s.*2;
    %y = dzdy.*sign(s);
end

end

