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
%face=convertToFormat(geometry,texture);
%drawFace(face);
%Let's scale it so things work out
% geometry_1e3=geometry/1e3;
[vertices,faces] = cybconvert(geometry);
vertices = reshape(vertices,[512*512,3]);
%%Calculate faces and vertices so we can render this in blender
% [faces,vertices]=surf2patch(cyb_vertices(:,:,1),cyb_vertices(:,:,2),cyb_vertices(:,:,3),'triangles');
%save('JUBF234.mat');
save('JUHF248.mat');

%%Patching shows you how to render in the cylidrical coordinate space but
%%it is a flat map and we really need a mesh interpolation,vertex,surface
%cyb_vertices = reshape(cyb_vertices,[512*512,3]);
%patch(cyb_vertices(:,1),cyb_vertices(:,2),cyb_vertices(:,3))
