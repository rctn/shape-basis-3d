import blenderLoad
import os

bl = blenderLoad.blenderLoad()
data_path = os.getenv('DATA')

files = os.listdir(data_path+'3dFace/matfiles/')

for ii in range(len(files)):
    fname = data_path + '3dFace/matfiles/'+files[ii]
    bpy = bl.load_face(fname)
    bpy = bl.generate_data(bpy,files[ii],1)
