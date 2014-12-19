%%

geometry = cybread('JUBF234');
texture = imread('JUBF234.png');
texture = permute(texture,[2,1,3]);
texture = texture(512:-1:1,:,:);
face=convertToFormat(geometry,texture);
drawFace(face);