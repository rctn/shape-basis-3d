%Script to write out a ply file with color information
function [err] = write_ply(fname,fvc)
err=1;
%File Handle
fid=fopen(fname,'w');
faces=fvc.faces;
x=fvc.vertices(:,1);
y=fvc.vertices(:,2);
z=fvc.vertices(:,3);

r=fvc.facevertexcdata(:,1);
g=fvc.facevertexcdata(:,2);
b=fvc.facevertexcdata(:,3);



%Assumes that the length of indices is exact for all inputs
fprintf(fid,'ply\n');
fprintf(fid,'format ascii 1.0\n');
fprintf(fid,'comment author : Mayur Mudigonda\n');
fprintf(fid,'element vertex %d\n',length(x));
fprintf(fid,'property float x\n');
fprintf(fid,'property float y\n');
fprintf(fid,'property float z\n');
fprintf(fid,'property uchar red\n');
fprintf(fid,'property uchar green\n');
fprintf(fid,'property uchar blue\n');
fprintf(fid,'element face %d\n',size(faces,1));
fprintf(fid,'property list uchar int vertex_index\n');
fprintf(fid,'end_header\n');
for ii=1:length(x)     
    fprintf(fid,'%f %f %f %d %d %d\n',x(ii),y(ii),z(ii),r(ii),g(ii),b(ii));
end

for ii=1:size(faces,1)
   fprintf(fid,'%d %d %d %d %d\n',3,faces(ii,1),faces(ii,2),faces(ii,3));
end
%Close File
fclose(fid);

err=0

end