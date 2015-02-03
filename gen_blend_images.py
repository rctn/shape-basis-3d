import sys
sys.path.append('/home/mudigonda/blender-git/build_opencolorio/bin')
import bpy
import numpy as np

fov=100
scene = bpy.context.scene
#scene.render.image_settings.file_fomat = 'jpg'
scene.render.resolution_x = 512
scene.render.resolution_y = 512

scene.camera.data.angle = fov*(np.pi/180.0)
num_samples=1000
switch='self'
np.random.seed(1)

for object in bpy.data.objects:
    if object.name != 'Lamp' and object.name!= 'Camera':
        object.select = True
bpy.ops.object.delete()
bpy.ops.mesh.primitive_ico_sphere_add()
bpy.types.ImageFormatSettings.color_mode='BW'

for ii in range(1000):
    xx=np.random.random(1)
    yy=np.random.random(1)
    if switch=='world': # moving object
        bpy.data.objects['Icosphere'].location = (xx,yy,0.0)
        path = '/media/mudigonda/Gondor/Data/sensorimotor/world/'+ 'sphere_'+str(xx)+'_'+str(yy)+'_'+str(0.0)+'.jpg'
    elif switch=='self': #moving self
        cam_location = bpy.data.objects['Camera'].location
        if np.random.random(1) > 0.5:
            tmp = np.asarray(cam_location) + np.asarray([xx,yy,0.0])
            bpy.data.objects['Camera'].location = tuple(tmp) 
        else:
            tmp = np.asarray(cam_location) - np.asarray([xx,yy,0.0])
            bpy.data.objects['Camera'].location = tuple(tmp) 
        path = '/media/mudigonda/Gondor/Data/sensorimotor/self/'+ 'sphere_'+str(xx)+'_'+str(yy)+'_'+str(0.0)+'.jpg'
    elif switch=='both': #moving
        path = '/media/mudigonda/Gondor/Data/sensorimotor/both/'+ 'sphere_'+str(xx)+'_'+str(yy)+'_'+str(0.0)+'.jpg'
    bpy.data.scenes['Scene'].render.filepath = path
    bpy.ops.render.render(write_still=True)
