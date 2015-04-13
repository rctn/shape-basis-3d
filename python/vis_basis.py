#Fuck it let's start new
import scipy.io as scio
import sys
import numpy as np
sys.path.append('//home/mudigonda/blender-git/build_opencolorio/bin/')
import bpy
from math import radians
import random
import pdb
import os

#jubf234=scio.loadmat('/media/mudigonda/Gondor/Projects/shape-basis-3d/matlab/JUBF234.mat')
class blenderLoad:
        def __init__(self):
            return 

        def load_face(self,fname,savepath,texture_flag=0):
            jubf234=scio.loadmat(fname)
            try:
                faces=jubf234['faces']
            except:
                pdb.set_trace()
            faces = faces-1
            vertices=jubf234['vertices']
            if texture_flag == 1: 
                texture=jubf234['texture']
            else:
                print('No texture map detected. So going to render only shape info')
                texture_flag = 0

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
            #creating a new object to link the object
            obj = bpy.data.objects.new("My_Object", mesh_3d)
            #dm = obj.modifiers.new('Decimate','DECIMATE')
            #dm.ratio = 0.5

            #we take the context
            scene = bpy.context.scene
            #link the object to the scene
            scene.objects.link(obj)

            if texture_flag == 1:
                vertex_colors = mesh_3d.vertex_colors

                if len(vertex_colors)==0:
                    vertex_colors.new()

                color_layer = vertex_colors['Col']
                i = 0
                non_zero = np.nonzero(texture)


                face_flat = faces.flatten()
                for ii in range(face_flat.shape[0]):
                    color_layer.data[ii].color = list(texture[face_flat[ii],:]/255.0)

                mat = bpy.data.materials.new('vertex_material')
                mat.use_vertex_color_paint = True
                mat.use_vertex_color_light = True
                mesh_3d.materials.append(mat)

            for object in bpy.data.objects:
                if object.name != 'Lamp' and object.name !='Camera':
                    object.select = True
                    bpy.context.scene.objects.active = object

            print(bpy.context.selected_objects)
            myObj = bpy.context.selected_objects[0]
            print(myObj)
            #Not sure what these next pair of statements do but they are important to edit object
            bpy.ops.object.mode_set(mode='OBJECT')
            

            ##Resize
            myObj.scale = ((0.05,0.05,0.05))
            #myObj.scale = ((0.07,0.07,0.07))
            ##Translate
            bpy.data.objects["My_Object"].location=(0.0,0.0,0.0)

            ##camera
            ##Location
            bpy.data.objects['Camera'].location = (0.0,0.0,15.0)
            #Rotation
            bpy.data.objects['Camera'].rotation_euler = (radians(0.0),radians(0.0),radians(0.0))

            ##Move the Lamp aka Light
            bpy.data.objects["Lamp"].location = (0.0,0.0,15.0)


            #FOV
            fov=75
            #FOV
            bpy.context.scene.camera.data.angle = fov*(np.pi/180.0)

            #Turn off the shader so things don't seem so specular
            for item in bpy.data.materials:
                item.use_shadeless=False

            #Rendering file
            bpy.context.scene.render.resolution_x=512
            bpy.context.scene.render.resolution_y=512
            bpy.types.ImageFormatSettings.color_mode='RGB'
            bpy.data.scenes['Scene'].render.filepath = savepath
            bpy.ops.render.render(write_still=True)
            print('Rendering Complete')
            return 
