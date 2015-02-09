%Load a few faces
DATA=getenv('DATA');
fpath=strcat(DATA,'3dFace/geometry');
files=dir(fpath);

for ii=4:length(files)
		%Do RTRIM to get only the file name
    display(files(ii).name)
		%Pass that forward
    load_data(files(ii).name);
end
