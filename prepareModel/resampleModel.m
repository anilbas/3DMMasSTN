function [ newmodel ] = resampleModel( shapeMU,shapePC,shapeEV,tl,UV,res )
%RESAMPLEMODEL Resample BFM so that vertices map to regular grid in UV 
%   This function resamples the BFM so that the vertices can be mapped to a
%   regular grid in UV space simply by reshaping. The grid is of dimension
%   res x res and UV contains the UV coordinates for the original model.

shapeMU = double(shapeMU);
shapePC = double(shapePC);

vertices = reshape(shapeMU,3,53490)';

[gridx,gridy]=meshgrid(0:1/(res-1):1,0:1/(res-1):1);
% Transpose rows and columns to be consistent with column-major ordering of
% Matconvnet STN code
gridx = gridx';
gridy = flipud(gridy)'; % Flip so that right way up in STN code

%gridx = gridx';
%gridy =(gridy)'; % Flip so that right way up in STN code

% There seems to be a bug with pointLocation - points that lie on a
% triangle edge on the boundary of the mesh (or the symmetry line) are 
% given NaN triangle index
% Hack to fix for now: shift the boundary outwards by eps
UV(UV(:,1)==0,1)=-eps;
UV(UV(:,1)==1,1)=1+eps;
UV(UV(:,2)==0,2)=-eps;
UV(UV(:,2)==1,2)=1+eps;
% Shift symmetry line by eps
if mod(res,2)~=0
    UV(UV(:,1)==0.5,1)=0.5+eps;
end

TR = triangulation(tl, UV(:,1), UV(:,2));

[TI, BC] = pointLocation(TR, gridx(:),gridy(:));

%texture = double(reshape(texMU,3,53490)')./255;

%im(:,:,1) = reshape(texture(tl(TI,1),1).*BC(:,1)+texture(tl(TI,2),1).*BC(:,2)+texture(tl(TI,3),1).*BC(:,3),res,res);
%im(:,:,2) = reshape(texture(tl(TI,1),2).*BC(:,1)+texture(tl(TI,2),2).*BC(:,2)+texture(tl(TI,3),2).*BC(:,3),res,res);
%im(:,:,3) = reshape(texture(tl(TI,1),3).*BC(:,1)+texture(tl(TI,2),3).*BC(:,2)+texture(tl(TI,3),3).*BC(:,3),res,res);

newverts(:,1) = vertices(tl(TI,1),1).*BC(:,1)+vertices(tl(TI,2),1).*BC(:,2)+vertices(tl(TI,3),1).*BC(:,3);
newverts(:,2) = vertices(tl(TI,1),2).*BC(:,1)+vertices(tl(TI,2),2).*BC(:,2)+vertices(tl(TI,3),2).*BC(:,3);
newverts(:,3) = vertices(tl(TI,1),3).*BC(:,1)+vertices(tl(TI,2),3).*BC(:,2)+vertices(tl(TI,3),3).*BC(:,3);

count = 0;
for i=1:res-1
    for j=1:res-1
        count = count+1;
        newfaces(count,1)=sub2ind([res res],i,j);
        newfaces(count,3)=sub2ind([res res],i,j+1);
        newfaces(count,2)=sub2ind([res res],i+1,j);
        count = count+1;
        newfaces(count,1)=sub2ind([res res],i+1,j+1);
        newfaces(count,2)=sub2ind([res res],i,j+1);
        newfaces(count,3)=sub2ind([res res],i+1,j);
    end
end

for dim = 1:size(shapePC,2)
    PC = shapePC(:,dim);
    PC = reshape(PC,3,53490)';
    PCnew(:,1) = PC(tl(TI,1),1).*BC(:,1)+PC(tl(TI,2),1).*BC(:,2)+PC(tl(TI,3),1).*BC(:,3);
    PCnew(:,2) = PC(tl(TI,1),2).*BC(:,1)+PC(tl(TI,2),2).*BC(:,2)+PC(tl(TI,3),2).*BC(:,3);
    PCnew(:,3) = PC(tl(TI,1),3).*BC(:,1)+PC(tl(TI,2),3).*BC(:,2)+PC(tl(TI,3),3).*BC(:,3);
    newshapePC(:,dim) = reshape(PCnew',3*res*res,1); %.*shapeEV(dim);
    %norms(dim) = norm(newshapePC(:,dim));
    %newshapePC(:,dim) = newshapePC(:,dim)./norms(dim);
end

%[U,S,V]=svd(newshapePC);
%newshapePC = U*[eye(size(V,1)); zeros(size(U,1)-size(V,1),size(V,1))]*V';

%for dim = 1:size(shapePC,2)
%    newshapePC(:,dim) = newshapePC(:,dim).*norms(dim);
%end

newmodel.nverts = res*res;
newmodel.shapeMU = reshape(newverts',newmodel.nverts*3,1);
newmodel.faces = newfaces;
newmodel.shapePC = newshapePC;

end

