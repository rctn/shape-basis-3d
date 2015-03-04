%Creating three different types of visualizations of the face for shape-PCA
fname='ADCM370'
close all;

[vertices,faces,geometry,texture]=load_data(fname);

%Figure1 -- Do nothing
h1=figure;
imagesc(geometry);
colormap gray;
saveas(h1,'~/tmp/squishedout.png');

h2=figure;
tmp=vertices(:,3);
imagesc(reshape(tmp,[512,512]));
colormap gray;
saveas(h2,'~/tmp/less_squishedout.png');

h3=figure;
tmp1=vertices(:,1);
tmp2=vertices(:,2);
mesh(reshape(tmp1,[512,512]),reshape(tmp2,[512,512]),reshape(tmp,[512,512]));
colormap gray;
axis off;
view(0,90);
saveas(h3,'~/tmp/shouldberight.png');