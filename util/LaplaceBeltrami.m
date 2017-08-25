function [Delta,Grad]=LaplaceBeltrami(X,F)
%Adapted from the tutorial of Gabriel Peyre:
%http://www.numerical-tours.com/matlab/meshproc_7_geodesic_poisson/

n = size(X,2);
m = size(F,2);

% Callback to get the coordinates of all the vertex of index i=1,2,3 in all faces
XF = @(i)X(:,F(i,:));

% Compute un-normalized normal through the formula e1xe2 where ei are the edges.
Na = cross( XF(2)-XF(1), XF(3)-XF(1) );

% Compute the area of each face as half the norm of the cross product.
amplitude = @(X)sqrt( sum( X.^2 ) );
A = amplitude(Na)/2;

% Compute the set of unit-norm normals to each face.
normalize = @(X)X ./ repmat(amplitude(X), [3 1]);
N = normalize(Na);

% Populate the sparse entries of the matrices for the operator implementing ?i?fui(Nf?ei)
I = []; J = []; V = []; % indexes to build the sparse matrices
for i=1:3
    % opposite edge e_i indexes
    s = mod(i,3)+1;
    t = mod(i+1,3)+1;
    % vector N_f^e_i
    wi = cross(XF(t)-XF(s),N);
    % update the index listing
    I = [I, 1:m];
    J = [J, F(i,:)];
    V = [V, wi];
end

% Sparse matrix with entries 1/(2Af)
dA = spdiags(1./(2*A(:)),0,m,m);

% Compute gradient.
GradMat = {};
for k=1:3
    GradMat{k} = dA*sparse(I,J,V(k,:),m,n);
end

% ? gradient operator.
Grad = @(u)[GradMat{1}*u, GradMat{2}*u, GradMat{3}*u]';

% Compute divergence matrices as transposed of grad for the face area inner product.
dAf = spdiags(2*A(:),0,m,m);
DivMat = {GradMat{1}'*dAf, GradMat{2}'*dAf, GradMat{3}'*dAf};

% Div operator.
%Div = @(q)DivMat{1}*q(1,:)' + DivMat{2}*q(2,:)' + DivMat{3}*q(3,:)';

%Laplacian operator as the composition of grad and div.
Delta = DivMat{1}*GradMat{1} + DivMat{2}*GradMat{2} + DivMat{3}*GradMat{3};

