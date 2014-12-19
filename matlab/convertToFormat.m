function face_format = convertToFormat(rangeData, texture)

%%  DESCRIPTION
%
%       Reads in cyberware range data and texture matrix, returns 6 column
%       rendering format [x,y,z,r,g,b]
%
%%  INPUT PARAMETERS
%
%       rangeData = the matrix containing face geometry data in cylindrical
%       coordinate space
%       
%%  OUTPUT PARAMETER
%
%       face_format = the 6 column rendering format
%%

CULLING_DISTANCE = 10000; % distance from centre in micro meters at which a pixel must be not to be culled

[c, r] = find(rangeData' > CULLING_DISTANCE); % cull invalid pixels         
pixel_order = [r,c];   

geometry = cybconvert(rangeData); % convert geometry to euclideon space

face_format = zeros(size(pixel_order, 1), 6);   % allocate memory for 6 column format matrix

for m=1:size(pixel_order, 1);
    % geometry
    face_format(m,1) = geometry(pixel_order(m,1), pixel_order(m,2), 1);      
    face_format(m,2) = geometry(pixel_order(m,1), pixel_order(m,2), 2);        
    face_format(m,3) = geometry(pixel_order(m,1), pixel_order(m,2), 3);
    % texture
    face_format(m,4) = texture(pixel_order(m,1), pixel_order(m,2), 1);
    face_format(m,5) = texture(pixel_order(m,1), pixel_order(m,2), 2);   
    face_format(m,6) = texture(pixel_order(m,1), pixel_order(m,2), 3);      
end

end


