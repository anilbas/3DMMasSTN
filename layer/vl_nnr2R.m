function [ R ] = vl_nnr2R( r,dCdR )
%r2R Axis angle to rotation matrix layer
% Forwards mode:
% R = vl_nnr2R(r);
%   r is of size 1 x 1 x 3 x nbatch containing axis angle vectors
%   R is of size 1 x 3 x 3 x nbatch containing rotation matrices
% Backwards mode:
% dCdr = vl_nnr2R(r,dCdR);
%   dCdR is derivative of the cost function with respect to R and is of
%   size 1 x 3 x 3 x nbatch
%   dCdr is derivative of the cost function with respect to r and is of
%   size 1 x 1 x 3 x nbatch
%
% Useful reference for formula:
% [1] Gallego, Guillermo, and Anthony Yezzi. "A compact formula for the 
%     derivative of a 3-D rotation in exponential coordinates." Journal 
%     of Mathematical Imaging and Vision 51.3 (2015): 378-384.

nbatch = size(r,4);
if nargin<2
    % Forwards

    % Vectors with zero length are a special case - they just have an
    % identity rotation matrix
    idmask = squeeze(all(r==0,3));

    R = zeros(1,3,3,nbatch,'single');
    
    % Fill in the identity cases
    R(1,1,1,idmask)=1;
    R(1,2,2,idmask)=1;
    R(1,3,3,idmask)=1;

    theta = sqrt(sum(r(:,:,:,~idmask).^2,3)); % 1 x 1 x 1 x non-zero nbatch rotation angles
    k1 = r(:,:,1,~idmask)./theta; % rotation axis unit vectors
    k2 = r(:,:,2,~idmask)./theta; % rotation axis unit vectors
    k3 = r(:,:,3,~idmask)./theta; % rotation axis unit vectors
    
    % Fill in the non-identity cases using the following formula (see [1]):
    % K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0]; 
    % R = eye(3) + sin(theta).*K + (1-cos(theta)).*K*K;
    
    R(1,1,1,~idmask) = (cos(theta) - 1).*k2.^2 + (cos(theta) - 1).*k3.^2 + 1;
    R(1,1,2,~idmask) =  -k3.*sin(theta) - k1.*k2.*(cos(theta) - 1);
    R(1,1,3,~idmask) = k2.*sin(theta) - k1.*k3.*(cos(theta) - 1);
    R(1,2,1,~idmask) = k3.*sin(theta) - k1.*k2.*(cos(theta) - 1);
    R(1,2,2,~idmask) = (cos(theta) - 1).*k1.^2 + (cos(theta) - 1).*k3.^2 + 1;
    R(1,2,3,~idmask) = -k1.*sin(theta) - k2.*k3.*(cos(theta) - 1);
    R(1,3,1,~idmask) = -k2.*sin(theta) - k1.*k3.*(cos(theta) - 1);
    R(1,3,2,~idmask) = k1.*sin(theta) - k2.*k3.*(cos(theta) - 1);
    R(1,3,3,~idmask) = (cos(theta) - 1).*k1.^2 + (cos(theta) - 1).*k2.^2 + 1;
else
    % Backwards
    
	idmask = squeeze(all(r==0,3));
    
    % Handle identity case
    dRdr1 = zeros(size(dCdR));
    dRdr1(1,2,3,idmask)=-1;
    dRdr1(1,3,2,idmask)=1;
    dRdr2 = zeros(size(dCdR));
    dRdr2(1,1,3,idmask)=1;
    dRdr2(1,3,1,idmask)=-1;
    dRdr3 = zeros(size(dCdR));
    dRdr3(1,1,2,idmask)=-1;
    dRdr3(1,2,1,idmask)=1;
    
    % Call forwards version to get R
    [ R ] = vl_nnr2R( r );
    
    % Subselect and precompute some values
    R11 = R(1,1,1,~idmask); R12 = R(1,1,2,~idmask); R13 = R(1,1,3,~idmask);
    R21 = R(1,2,1,~idmask); R22 = R(1,2,2,~idmask); R23 = R(1,2,3,~idmask);
    R31 = R(1,3,1,~idmask); R32 = R(1,3,2,~idmask); R33 = R(1,3,3,~idmask);
    r1 = r(1,1,1,~idmask); r2 = r(1,1,2,~idmask); r3 = r(1,1,3,~idmask);
    sqnorm = (r1.^2 + r2.^2 + r3.^2);
    
    % For all other cases, find 3x3 derivatives of R with respect to each
    % element of r
    dRdr1(1,1,1,~idmask) = (R31.*(R31.*r1 + r1.*r2 - r3.*(R11 - 1)))./sqnorm - (R21.*(r1.*r3 - R21.*r1 + r2.*(R11 - 1)))./sqnorm;
    dRdr1(1,1,2,~idmask) = (R32.*(R31.*r1 + r1.*r2 - r3.*(R11 - 1)))./sqnorm - (R22.*(r1.*r3 - R21.*r1 + r2.*(R11 - 1)))./sqnorm;
    dRdr1(1,1,3,~idmask) = (R33.*(R31.*r1 + r1.*r2 - r3.*(R11 - 1)))./sqnorm - (R23.*(r1.*r3 - R21.*r1 + r2.*(R11 - 1)))./sqnorm;
    dRdr1(1,2,1,~idmask) = (R11.*(r1.*r3 - R21.*r1 + r2.*(R11 - 1)))./sqnorm - (R31.*(r1.^2 + R21.*r3 - R31.*r2))./sqnorm;
    dRdr1(1,2,2,~idmask) = (R12.*(r1.*r3 - R21.*r1 + r2.*(R11 - 1)))./sqnorm - (R32.*(r1.^2 + R21.*r3 - R31.*r2))./sqnorm;
    dRdr1(1,2,3,~idmask) = (R13.*(r1.*r3 - R21.*r1 + r2.*(R11 - 1)))./sqnorm - (R33.*(r1.^2 + R21.*r3 - R31.*r2))./sqnorm;
    dRdr1(1,3,1,~idmask) = (R21.*(r1.^2 + R21.*r3 - R31.*r2))./sqnorm - (R11.*(R31.*r1 + r1.*r2 - r3.*(R11 - 1)))./sqnorm;        
    dRdr1(1,3,2,~idmask) = (R22.*(r1.^2 + R21.*r3 - R31.*r2))./sqnorm - (R12.*(R31.*r1 + r1.*r2 - r3.*(R11 - 1)))./sqnorm;
    dRdr1(1,3,3,~idmask) = (R23.*(r1.^2 + R21.*r3 - R31.*r2))./sqnorm - (R13.*(R31.*r1 + r1.*r2 - r3.*(R11 - 1)))./sqnorm;
    
    dRdr2(1,1,1,~idmask) = (R31.*(r2.^2 - R12.*r3 + R32.*r1))./sqnorm - (R21.*(R12.*r2 + r2.*r3 - r1.*(R22 - 1)))./sqnorm;
    dRdr2(1,1,2,~idmask) = (R32.*(r2.^2 - R12.*r3 + R32.*r1))./sqnorm - (R22.*(R12.*r2 + r2.*r3 - r1.*(R22 - 1)))./sqnorm;
    dRdr2(1,1,3,~idmask) = (R33.*(r2.^2 - R12.*r3 + R32.*r1))./sqnorm - (R23.*(R12.*r2 + r2.*r3 - r1.*(R22 - 1)))./sqnorm;
    dRdr2(1,2,1,~idmask) = (R11.*(R12.*r2 + r2.*r3 - r1.*(R22 - 1)))./sqnorm - (R31.*(r1.*r2 - R32.*r2 + r3.*(R22 - 1)))./sqnorm;
    dRdr2(1,2,2,~idmask) = (R12.*(R12.*r2 + r2.*r3 - r1.*(R22 - 1)))./sqnorm - (R32.*(r1.*r2 - R32.*r2 + r3.*(R22 - 1)))./sqnorm;
    dRdr2(1,2,3,~idmask) = (R13.*(R12.*r2 + r2.*r3 - r1.*(R22 - 1)))./sqnorm - (R33.*(r1.*r2 - R32.*r2 + r3.*(R22 - 1)))./sqnorm;
    dRdr2(1,3,1,~idmask) = (R21.*(r1.*r2 - R32.*r2 + r3.*(R22 - 1)))./sqnorm - (R11.*(r2.^2 - R12.*r3 + R32.*r1))./sqnorm;
    dRdr2(1,3,2,~idmask) = (R22.*(r1.*r2 - R32.*r2 + r3.*(R22 - 1)))./sqnorm - (R12.*(r2.^2 - R12.*r3 + R32.*r1))./sqnorm;
    dRdr2(1,3,3,~idmask) = (R23.*(r1.*r2 - R32.*r2 + r3.*(R22 - 1)))./sqnorm - (R13.*(r2.^2 - R12.*r3 + R32.*r1))./sqnorm;

    dRdr3(1,1,1,~idmask) = (R31.*(r2.*r3 - R13.*r3 + r1.*(R33 - 1)))./sqnorm - (R21.*(r3.^2 + R13.*r2 - R23.*r1))./sqnorm;
    dRdr3(1,1,2,~idmask) = (R32.*(r2.*r3 - R13.*r3 + r1.*(R33 - 1)))./sqnorm - (R22.*(r3.^2 + R13.*r2 - R23.*r1))./sqnorm;
    dRdr3(1,1,3,~idmask) = (R33.*(r2.*r3 - R13.*r3 + r1.*(R33 - 1)))./sqnorm - (R23.*(r3.^2 + R13.*r2 - R23.*r1))./sqnorm;
	dRdr3(1,2,1,~idmask) = (R11.*(r3.^2 + R13.*r2 - R23.*r1))./sqnorm - (R31.*(R23.*r3 + r1.*r3 - r2.*(R33 - 1)))./sqnorm;
    dRdr3(1,2,2,~idmask) = (R12.*(r3.^2 + R13.*r2 - R23.*r1))./sqnorm - (R32.*(R23.*r3 + r1.*r3 - r2.*(R33 - 1)))./sqnorm;
    dRdr3(1,2,3,~idmask) = (R13.*(r3.^2 + R13.*r2 - R23.*r1))./sqnorm - (R33.*(R23.*r3 + r1.*r3 - r2.*(R33 - 1)))./sqnorm;
    dRdr3(1,3,1,~idmask) = (R21.*(R23.*r3 + r1.*r3 - r2.*(R33 - 1)))./sqnorm - (R11.*(r2.*r3 - R13.*r3 + r1.*(R33 - 1)))./sqnorm;
    dRdr3(1,3,2,~idmask) = (R22.*(R23.*r3 + r1.*r3 - r2.*(R33 - 1)))./sqnorm - (R12.*(r2.*r3 - R13.*r3 + r1.*(R33 - 1)))./sqnorm;
    dRdr3(1,3,3,~idmask) = (R23.*(R23.*r3 + r1.*r3 - r2.*(R33 - 1)))./sqnorm - (R13.*(r2.*r3 - R13.*r3 + r1.*(R33 - 1)))./sqnorm;

    dCdr(1,1,1,:) = sum(sum(dCdR.*dRdr1,2),3);
    dCdr(1,1,2,:) = sum(sum(dCdR.*dRdr2,2),3);
    dCdr(1,1,3,:) = sum(sum(dCdR.*dRdr3,2),3);
        
    R = dCdr;
end

end