% Load original BFM
load('01_MorphableModel.mat');

% Load modified FW expression model
load('3DDFA_Release/Matlab/Model_Expression.mat');

% Load FW to BFM mapping
load('map_tddfa_to_basel.mat');
% Fix zero-based indexing
map_tddfa_to_basel=map_tddfa_to_basel+1;

model.shapePC = zeros(3*53490,10);
for i=1:5
model.shapePC(:,i) = shapePC(:,i).*shapeEV(i);
end

%% Extrapolate expressions to mouth interior via Laplace-Beltrami
vertices = double(reshape(shapeMU,3,53490)');
[L,~]=LaplaceBeltrami(vertices',tl');
% Selection matrix for BFM vertices to FW
S = sparse(1:length(map_tddfa_to_basel),double(map_tddfa_to_basel),ones(length(map_tddfa_to_basel),1),length(map_tddfa_to_basel),53490);
for i=1:5
    expression = ([L; S]\[L*vertices; S*vertices+reshape(w_exp(:,i),3,53215)'])-vertices;
    expression = expression';
    model.shapePC(:,i+5) = expression(:);
end
%% Load UV coordinates and resample model
load('BFM_UV.mat');
[ newmodel ] = resampleModel( shapeMU,model.shapePC,[],tl,UV,112 );
newmodel.shapeMU=newmodel.shapeMU./1000;
newmodel.shapePC=newmodel.shapePC./1000;
%%
clear FV
FV.faces = newmodel.faces;
alpha = randn(10,1);
FV.vertices = reshape(newmodel.shapeMU+newmodel.shapePC*alpha,3,112^2)';
figure; patch(FV, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong'); axis equal; light; axis tight