#Fuck it let's start new
import scipy.io as scio
import sys
import numpy as np
sys.path.append('//home/mudigonda/blender-git/build_opencolorio/bin/')
import bpy
from math import radians
import random

jubf234=scio.loadmat('/media/mudigonda/Gondor/Projects/shape-basis-3d/matlab/JUBF234.mat')
faces=jubf234['faces']
vertices=jubf234['vertices']
texture=jubf234['texture']

for object in bpy.data.objects:
    if object.name != 'Lamp' and object.name !='Camera':
        object.select = True
objs = [l for l in bpy.data.objects]
print(objs)

#Deleting existing object
bpy.ops.object.delete()

#Creating a new mesh
mesh_3d= bpy.data.meshes.new("Human_Face")

#converting stuff to int and making a list out of a numpy array...fuuuuuck
#Remember that these are matlab indices, somethign to worry about
vertex=[]
for ii in range(vertices.shape[0]):
    vertex.append((float(vertices[ii,0]),float(vertices[ii,1]),float(vertices[ii,2])))

#converting more stuff to int, same thing except here out of face data where face == triangles
face=[]
for ii in range(faces.shape[0]):
   face.append((int(faces[ii,0]),int(faces[ii,1]),int(faces[ii,2])))

#This lets us create our own mesh of an object
mesh_3d.from_pydata(vertex, [], face)  
#We update the scene
mesh_3d.update()


#We take the vertex colors
vertex_colors = mesh_3d.vertex_colors

if len(vertex_colors)==0:
    vertex_colors.new()

color_layer = vertex_colors['Col']
i = 0
print("Counting nonzero Texture elements")
print(np.count_nonzero(texture))
for poly in mesh_3d.polygons:
    for idx in poly.loop_indices:
        rgb = [random.random() for i in range(3)]
#        color_layer.data[i].color = rgb
#        if i<262144:
        color_layer.data[i].color = rgb
#            color_layer.data[i].color = texture[i,:]/255.0
#        else:
        i += 1
#            print(i)

mat = bpy.data.materials.new('vertex_material')
mat.use_vertex_color_paint = True
mat.use_vertex_color_light = True
mesh_3d.materials.append(mat)
mesh_3d.update()

#creating a new object to link the object
obj = bpy.data.objects.new("My_Object", mesh_3d)  

#we take the context
scene = bpy.context.scene    
#link the object to the scene
scene.objects.link(obj)    
#we select the object, we want to work with objects over meshes. that be the idea
obj.select = True    

#Experimenting with trying to set active object, seems more or less useless or a duplicate of the previous statements
for object in bpy.data.objects:
    if object.name != 'Lamp' and object.name !='Camera':
        object.select = True
        bpy.context.scene.objects.active = object

objs = [l for l in bpy.data.objects]
print(objs)
print(bpy.context.selected_objects)

myObj = bpy.context.selected_objects[0]
print(myObj)
#Not sure what these next pair of statements do but they are important to edit object
bpy.ops.object.mode_set(mode='OBJECT')
#bpy.ops.object.mode_set(mode='EDIT')
#bpy.ops.mesh.select_all(action='SELECT')

##Resize
#myObj.scale = ((0.025,0.025,0.025))
myObj.scale = ((0.05,0.05,0.05))
##Translate
bpy.data.objects["My_Object"].location=(0.0,0.0,0.0)

##camera
##Location
bpy.data.objects['Camera'].location = (0.0,0.0,10.0)
#Rotation
bpy.data.objects['Camera'].rotation_euler = (radians(0.0),radians(0.0),radians(0.0))

##Move the Lamp aka Light
bpy.data.objects["Lamp"].location = (0.0,0.0,15.0)


fov=150
#FOV
scene.camera.data.angle = fov*(np.pi/180.0)

#Rendering file
path = "load_face2.png" 
bpy.data.scenes['Scene'].render.filepath = path
bpy.ops.render.render(write_still=True)
