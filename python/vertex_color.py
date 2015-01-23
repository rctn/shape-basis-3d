import sys
sys.path.append('/home/mudigonda/blender-git/build_opencolorio/bin')
import bpy
import random
import numpy as np

# start in object mode
for object in bpy.data.objects:
    if object.name != 'Lamp' and object.name!= 'Camera':
        object.select = True

objs  = [l for l in bpy.data.objects]
print(objs)

bpy.ops.object.delete()
bpy.ops.mesh.primitive_cube_add()
#bpy.ops.material.new()
#Enable vertex colors in rendering
#bpy.data.materials["Material.001"].use_vertex_color_paint = True

# set to vertex paint mode to see the result
#bpy.ops.object.mode_set(mode='VERTEX_PAINT')

obj  = bpy.data.objects["Cube"]
mesh = obj.data
vertex_colors = mesh.vertex_colors

if len(vertex_colors)==0:
    vertex_colors.new()

"""
let us assume for sake of brevity that there is now 
a vertex color map called  'Col'    
"""

color_layer = vertex_colors['Col']

# or you could avoid using the color_layer name
# color_layer = mesh.vertex_colors.active  

i = 0
for poly in mesh.polygons:
    for idx in poly.loop_indices:
        rgb = [random.random() for i in range(3)]
        color_layer.data[i].color = rgb
        i += 1

mat = bpy.data.materials.new('vertex_material')
mat.use_vertex_color_paint = True
mat.use_vertex_color_light = True
mesh.materials.append(mat)

#obj.select = True
#bpy.context.space_data.context='MATERIAL'
#bpy.context.object.active_material.use_vertex_color_paint = True

#Set in the Solid Texture Mode
#bpy.context.space_data.show_textured_solid=True


path='/home/mudigonda/tmp/color_test.png'
print(path)
fov=50
scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.camera.data.angle = fov*(np.pi/180.0)
#bpy.context.scene.update()
bpy.types.ImageFormatSettings.color_mode='RGB'
bpy.data.scenes['Scene'].render.filepath=path
bpy.ops.render.render(write_still=True)
