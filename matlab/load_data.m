function [vertices,faces] = load_data(fname)
%Get data
DATA=getenv('DATA');

%Load Geometry
geometry = cybread(strcat(DATA,'3dFace/geometry/',fname));
%Load texture
texture = imread(strcat(DATA,'3dFace/textures/',fname,'.png'));
texture = permute(texture,[2,1,3]);
texture = texture(512:-1:1,:,:);
texture = reshape(texture,[512*512,3]);

[vertices]=cybconvert(geometry);
[faces,vertices]=surf2patch(vertices(:,:,1),vertices(:,:,2),vertices(:,:,3),'triangles');

save(fname);

end