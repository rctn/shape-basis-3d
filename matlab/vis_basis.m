%Function that takes a basis.mat file
%Converts from cylindrical coordinate spaces to X,Y,Z
%In addition, also create faces (triangulation)
%Saves it out
function [vertices,faces,geometry,texture] = vis_basis(fname,savepath) 
%Get data
DATA=getenv('DATA');

%Load Geometry
loadmat(strcat(DATA,'3dFace/shape_basis/PCA/basis.mat'))

%Convert Mean Face
[vertices]=cybconvert(geometry);
[faces,vertices]=surf2patch(vertices(:,:,1),vertices(:,:,2),vertices(:,:,3),'triangles');

output_path='/media/mudigonda/Gondor/Data/3dFace/matfiles/';
save(strcat(output_path,fname));

end
