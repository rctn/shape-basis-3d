
%Rendering the images correctly because python is a bitch ass programming
%language
close all;
clear all;

DATA=getenv('DATA')
proj_path = strcat(DATA,'3dFace/shape_basis/PCA/');
load(strcat(proj_path,'basis.mat'));
% mean_face = rot90(mean_face);
% mean_face = flipud(mean_face);

mean_fig = figure();
lon=reshape(vertices(:,1),[512,512]);
lat=reshape(vertices(:,2),[512,512]);
mean_reshape = reshape(mean_face,[512,512]);
% image(lon,lat,mean_reshape)
% surf(lon,lat,mean_reshape,mean_reshape);
% mesh(lon,lat,mean_reshape);
% mesh(reshape(mean_face,[512,512]))
meshz(lon,lat,mean_reshape);
axis off;
axis image;
% colormap gray;
view(0,90);
% print(mean_fig,'-dpng','-r100',strcat(proj_path,'mean_face'));
% set(gcf'Position',[0,0,512,512])
% truesize(mean_fig);
% export_fig strcat(proj_path,'mean_face.png') -native
saveas(mean_fig,strcat(proj_path,'mean_face.png'));


%%Plotting Eig Vectors
for ii=1:50
   eig_fig = figure();
   mesh(lon,lat,reshape(shape_eig_vectors_face(ii,:),[512,512]));
   axis off;
   view(0,90);
   saveas(eig_fig,strcat(proj_path,'eig_face_',int2str(ii),'.png'));
end

eig_val_fig = figure();
plot(real(shape_eig_vals));
saveas(eig_val_fig,strcat(proj_path,'eig_vals.png'));