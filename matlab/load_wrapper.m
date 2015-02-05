%Load a few faces
face_name = {'JUBF234','JUHF248','JUKM326','JUSF235','JUSM342'}
for ii=1:length(face_name)
    load_data(face_name{ii});
end