close all;
clear all;

%%
DATA=getenv('DATA')
%geometry = cybread(strcat(DATA,'3dFace/geometry/GEO#1/','JUBF234'));
geometry = cybread(strcat(DATA,'3dFace/geometry/GEO#1/','JUHF248'));
%texture = imread(strcat(DATA,'3dFace/textures/ImageSet#3/','JUBF234.png'));
texture = imread(strcat(DATA,'3dFace/textures/ImageSet#3/','JUHF248.png'));
texture = permute(texture,[2,1,3]);
texture = texture(512:-1:1,:,:);
texture = reshape(texture,[512*512,3]);
%Let's scale it so things work out
[vertices] = cybconvert(geometry);
% vertices = reshape(vertices,[512*512,3]);
% % vertices = unique(vertices,'rows');
%%Calculate faces and vertices so we can render this in blender
[faces,vertices]=surf2patch(vertices(:,:,1),vertices(:,:,2),vertices(:,:,3),'triangles');
%save('JUBF234.mat');
save('JUHF248.mat');

%%Patching shows you how to render in the cylidrical coordinate space but
%%it is a flat map and we really need a mesh interpolation,vertex,surface
%cyb_vertices = reshape(cyb_vertices,[512*512,3]);
%patch(cyb_vertices(:,1),cyb_vertices(:,2),cyb_vertices(:,3))


%%Now removing non essential faces (the ones with zeros)
%%First reshape vertices
vertex_reshape = reshape(vertices,[512*512,3]);
%%Find the vertices that have zeros in them
zero_idx = find(vertex_reshape(:,3)==0);
%%Now for zero_idx in faces (rows), delete those faces
face_row_to_delete = 0;
for ii=1:size(zero_idx,1)
    [row, col] = find(faces==zero_idx(ii));
    face_row_to_delete = [face_row_to_delete;row];
end
face_row_to_delete = unique(face_row_to_delete);

%set diff
tot_face_idx = 1:size(faces,1);
faces_keep = setdiff(tot_face_idx,face_row_to_delete);
faces=faces(faces_keep);
