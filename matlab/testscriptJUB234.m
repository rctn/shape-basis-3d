%%
DATA=getenv('DATA');
geometry = cybread(strcat(DATA,'3dFace/geometry/GEO#1/','JUBF234'));
texture = imread(strcat(DATA,'3dFace/textures/ImageSet#3/','JUBF234.png'));
texture = permute(texture,[2,1,3]);
texture = texture(512:-1:1,:,:);
%face=convertToFormat(geometry,texture);
%drawFace(face);
%%Calculate faces and vertices so we can render this in blender
[faces,vertices]=surf2patch(geometry,'triangles');
save('JUBF234.mat');