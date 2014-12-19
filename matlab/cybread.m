function matrix = cybread(filename)

%%  DESCRIPTION
%
%       Reads in cyberware file format and stores as an m by n matrix of
%       range values.
%
%%  INPUT PARAMETERS
%
%       filename = name of cyberware file
%       
%%  OUTPUT PARAMETER
%
%       matrix = the cyberware radius values by latitude and longitude
%       incremenets
%%

    % open file
    file = fopen(filename);    
    % read and display ASCII file parameters
    for i=1:21
        line = fgets(file);
        disp(line); 
    end  
    
    % read big endian byte format, each radius consisting of two bytes
    image = fread(file, [512,512], '*int16', 0, 'b');
    % rotate image 90 degrees,  so faces are upright
    image = rot90(image,2);  
    matrix = int32(image);
    
    % close cyberware file    
    fclose(file);  
    
    % shift bits << 4 
    for i = 1:size(matrix,1)
        for j = 1:size(matrix,2)
            if (matrix(i,j) ~= -32768)
                matrix(i,j) = bitshift(uint32(matrix(i,j)), 4); % bit shift by 4 bits, as specified by the file
            else
                matrix(i,j) = 0; % set to 0
            end
        end       
    end
    
    
end