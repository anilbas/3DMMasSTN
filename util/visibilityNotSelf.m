function [ visibility ] = visibilityNotSelf( FV )
%VISIBILITYNOTSELF Approximate visibility testing only for self occlusions

V2 = FV.vertices;

% Get the triangle vertices
v1 = FV.faces(:, 1);
v2 = FV.faces(:, 2);
v3 = FV.faces(:, 3);

% Compute the edge vectors
e1s =  V2(v2, :) - V2(v1, :);
e2s =  V2(v3, :) - V2(v1, :);
e3s =  V2(v2, :) - V2(v3, :);

% Normalize the edge vectors
e1s_norm = e1s ./ repmat(sqrt(sum(e1s.^2, 2)), 1, 3);
e2s_norm = e2s ./ repmat(sqrt(sum(e2s.^2, 2)), 1, 3);
e3s_norm = e3s ./ repmat(sqrt(sum(e3s.^2, 2)), 1, 3);

% Compute the angles
angles(:, 1) = acos(sum(e1s_norm .* e2s_norm, 2));
angles(:, 2) = acos(sum(e3s_norm .* e1s_norm, 2));
angles(:, 3) = pi - (angles(:, 1) + angles(:, 2));

% Compute the triangle weighted normals
triangle_normals    = cross(e1s, e3s, 2);
w1_triangle_normals = triangle_normals .* repmat(angles(:, 1), 1, 3);
w2_triangle_normals = triangle_normals .* repmat(angles(:, 2), 1, 3);
w3_triangle_normals = triangle_normals .* repmat(angles(:, 3), 1, 3);

% Initialize the vertex normals
normals = zeros(size(V2, 1), 3);

normals(v1,:) = normals(v1,:) + w1_triangle_normals;
normals(v2,:) = normals(v2,:) + w2_triangle_normals;
normals(v3,:) = normals(v3,:) + w3_triangle_normals;

% Self-occlusions
visibility = normals(:, 3) >= 0;

end

