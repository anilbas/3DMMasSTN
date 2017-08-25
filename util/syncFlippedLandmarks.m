function [ xpnew ] = syncFlippedLandmarks( xp )

lookup = [6 5 4 3 2 1 12 11 10 9 8 7 17 16 15 14 13 20 19 18 21];
if size(xp,1)==2
    xpnew(1,:) = xp(1,lookup);
    xpnew(2,:) = xp(2,lookup);
else
    xpnew = xp(lookup);
end


end

