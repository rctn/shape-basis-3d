function [vertices,faces] = cybconvert(image)

%%  DESCRIPTION
%
%       This function takes the cyberware matrix of cylindrical coordsinates
%       and produces a set of vertices
%
%%  INPUT PARAMETERS
%
%       image = cyberware image
%       
%%  OUTPUT PARAMETER
%
%       vertices = set of vertices
%%

    lat_incr  = 0.615;          % mm per latitude increment, 615 um;
    long_incr = 0.012272;       % radian per long increment, 12272 urad;
    height = size(image,1);
    width  = size(image,2);
    vertices = zeros(height, width, 3);   
    count = 1;
    y = (lat_incr*height)/2;    % centre face on y = 0
    for i = 1:size(image,1)
        y = y - lat_incr;
        for j = 1:size(image,2)  
            if (image(i,j) > 0)
                radius = double(image(i,j)) /1000;   % convert radius to mm
                angle = 2*pi - ((j-1) * long_incr);    
                if (angle < pi/2)
                    x = sin(angle) * radius;
                    z = -(cos(angle) * radius);
                elseif (angle < pi)
                    angle = angle - pi/2;
                    x = cos(angle) * radius;
                    z = sin(angle) * radius;
                elseif (angle < 3*pi/2)
                    angle = angle - pi;
                    x = -(sin(angle) * radius);                
                    z = cos(angle) * radius;
                else
                    angle = angle - 3*pi/2;
                    x = -(cos(angle) * radius);
                    z = -(sin(angle) * radius);
                end
                vertices(i,j,1) = x;
                vertices(i,j,2) = y;
                vertices(i,j,3) = z; 
                count = count + 1;
            end
            
        end
    end

disp('Done converting format')
%Creating the faces (polygons)
faces=zeros((size(image,1)-1)*2,3);
kk=1;
for ii = 1:size(image,1)-1    
    for jj =1:size(image,2)-1
        %Index 1
%         faces(kk,1)= sub2ind([height,width],ii,jj);
        faces(kk,1)= sub2ind([height,width],jj,ii);
        %Index 2
%         faces(kk,2)= sub2ind([height,width],ii,jj+1);        
        faces(kk,2)= sub2ind([height,width],jj+1,ii);        
        %Index 3       
%         faces(kk,3)= sub2ind([height,width],ii+1,jj);
        faces(kk,3)= sub2ind([height,width],jj,ii+1);
        kk=kk+1;
        %Index 1
%         faces(kk,1)= sub2ind([height,width],ii+1,jj);
        faces(kk,1)= sub2ind([height,width],jj,ii+1);
        %Index 2
%         faces(kk,2)= sub2ind([height,width],ii,jj+1);        
        faces(kk,2)= sub2ind([height,width],jj+1,ii);        
        %Index 3       
%         faces(kk,3)= sub2ind([height,width],ii+1,jj+1);
        faces(kk,3)= sub2ind([height,width],jj+1,ii+1);
        kk=kk+1;
    end
end

end

