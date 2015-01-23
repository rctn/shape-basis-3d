#Fuck it let's start new
import scipy.io as scio
import sys
import numpy as np
sys.path.append('//home/mudigonda/blender-git/build_opencolorio/bin/')
import bpy
from math import radians

jubf234=scio.loadmat('/media/mudigonda/Gondor/Projects/shape-basis-3d/matlab/JUBF234.mat')
faces=jubf234['faces']
vertices=jubf234['vertices']

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
vertex=[]
for ii in range(vertices.shape[0]):
    vertex.append((float(vertices[ii,0]),float(vertices[ii,1]),float(vertices[ii,2])))

#converting more stuff to int, same thing except here out of face data where face == triangles
face=[]
for ii in range(faces.shape[0]):
   face.append((int(faces[ii,0])-1,int(faces[ii,1])-1,int(faces[ii,2])-1))

#This lets us create our own mesh of an object
mesh_3d.from_pydata(vertex, [], face)  
#We update the scene, which is like clicking render?
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

#Got this from the blender bpy documentation on how to override the context
#The idea is some operations require an explicit description of context while others dont
#Apparently we can know more by looking at what the operation polls()
#Unable to fix this at this point in time
for window in bpy.context.window_manager.windows:
    screen = window.screen
    print(screen.name)
    for area in screen.areas:
        print(area.type)
        if area.type == 'VIEW_3D':
            override = {'window':window,'screen':screen, 'area':area}
            #It crashes (segfaults) at the next statement
            #bpy.ops.screen.screen_full_area(override)
            break

#Not sure what these next pair of statements do but they are important to edit object
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.mode_set(mode='EDIT')

bpy.ops.mesh.select_all(action='SELECT')

##This works if you run it through blender player's python engine but not so when you run it through a standalone script
##rescale
#bpy.ops.transform.resize(value=(.025,.025,.025),constraint_axis=(False,False,False),constraint_orientation='GLOBAL',mirror=False,proportional='DISABLED',proportional_edit_falloff='SMOOTH',proportional_size=1)

#trying a less dense command in the hope that the defaults will cut it for us
bpy.ops.transform.resize(value=(.025,.025,.025))

#This fails
#bpy.data.objects["My_Object"].resize = (.025,.025,.025)

##This works if you run it through blender player's python engine but not so when you run it through a standalone script
#translate
#bpy.ops.transform.translate(value=(-5.0,-5.0,0.0),constraint_axis=(False,False,False),constraint_orientation='GLOBAL',mirror=False,proportional='DISABLED',proportional_edit_falloff='SMOOTH',proportional_size=1)
bpy.data.objects["My_Object"].location=(-5.0,-5.0,0.0)

#camera
#This works, since we used it previously
bpy.data.objects['Camera'].location = (0.0,5.0,5.0)
bpy.data.objects['Camera'].rotation_euler = (radians(90),0.0,0.0)


#Rendering file
path = "load_face2.png" 
bpy.data.scenes['Scene'].render.filepath = path
bpy.ops.render.render(write_still=True)
