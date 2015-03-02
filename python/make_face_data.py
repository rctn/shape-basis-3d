#creating lists of the file name
import string as str
import os
import numpy as np
import h5py
from scipy import ndimage
import shutil
import glob
import pdb
from subprocess import call
import random

def make_fname_lists(path='/media/mudigonda/Gondor/Data/3dFace/rotatedFaces/train/resized'):
    files = os.listdir(path)
    identity = []
    x_rot = []
    y_rot = []
    z_rot = []
    counter = 0
    for fname in files:
        if np.mod(counter,100)==0:
            print('The value of counter is %d',counter)

        split_filename = str.split(fname,'_')
        try:
            identity.append(split_filename[0])
            x_rot.append(int(split_filename[3]))
            y_rot.append(int(split_filename[6]))
            z_rot.append(int(str.split(split_filename[9],'.')[0]))
            counter = counter + 1
        except:
            print(split_filename)
            print("Something went wrong")
    print('Creating lists done')
    return files,identity,x_rot,y_rot,z_rot 

def write_hdf5(inputpath=None,outputfnames=None,no_of_files=None,write_flag=0):

    #First we create three file handles
    #Let's assume there are training folders and test folders
    if inputpath is None:
        inputpath='/media/mudigonda/Gondor/Data/3dFace/rotatedFaces/train/resized/'
    if outputfnames is None:
        outputfnames ='/media/mudigonda/Gondor/Data/3dFace/'
    if write_flag==0 :
        FLAG='train'
    else: 
        FLAG='test'

    try:
        [files,face_id,xx,yy,zz] = make_fname_lists(inputpath)
    except:
        print('Could not create file list')

    if no_of_files is None:
        no_of_files = len(files)

    print('Here are the parameters')
    print('Input Path',inputpath)
    print('Output Path',outputfnames)
    print('No of files',no_of_files)

    #Load a dummy file for getting image dimensions
    im_tmp = ndimage.imread(inputpath+files[0])
    size = im_tmp.shape
    print('Size of dummy image is',size)
    print('Trying to create training file handles')
    try:
        h5_input = h5py.File(outputfnames+FLAG+'_input'+'.hdf5',"w")
        h5_output = h5py.File(outputfnames+FLAG+'_output'+'.hdf5',"w")
        h5_transf = h5py.File(outputfnames+FLAG+'_transf.hdf5',"w")
        #make sure not to include the alpha channel, subtract 1
        inp_dset = h5_input.create_dataset("Input",(no_of_files*size[0]*size[1]*(size[2]-1),))
        op_dset = h5_output.create_dataset("Output",(no_of_files*size[0]*size[1]*(size[2]-1),))
        transf_dset = h5_transf.create_dataset("Transformation",(no_of_files*3,))
        INPUT_SIZE=size[0]*size[1]*(size[2]-1)
    except:
        print("It looks like I couldn't create the file handles for the training set")
        return 1

    #Then we iterate through the files list that we get tcalling making list
    print('Okay, here goes -- trying to write stuff into files')
    counter1 = 0
    counter2 = 0 
    #for ii in np.arange(len(face_id)):
    for ii in range(no_of_files):
        if np.mod(ii,1000)==0: 
            print('Written images = ',ii)
        #Read File
        try:
            im_ip = ndimage.imread(inputpath+files[ii])
            im_ip_wo_alpha = im_ip[:,:,0:3]
        except:
            print("Unable to read training input image")
        try:
            #For face chosen find all other entries of faces
            faces_idx = [i for i,x in enumerate(face_id) if x==face_id[ii]]
            #Generate random number
        except:
            print("Unable to find similar Faces. Fail",face_id[ii])
        try:
            rand_int = random.randint(0,len(faces_idx)-1) #weird bug it should not return len() but it does
        except:
            print("Failed to randint. Values of rand_int and faces_idx[rand_int] are",rand_int,faces_idx[rand_int]) 
        try:
            im_op = ndimage.imread(inputpath+files[faces_idx[rand_int]])
            im_op_wo_alpha = im_op[:,:,0:3]
        except:
            pdb.set_trace()
            print("Unable to load file, this is the file path",inputpath+files[faces_idx[rand_int]])
        try:
            #The transformation between the two is just Pose2-Pose1
            xx_diff = xx[faces_idx[rand_int]]-xx[ii]
            yy_diff = yy[faces_idx[rand_int]]-yy[ii]
            zz_diff = zz[faces_idx[rand_int]]-zz[ii]
        except:
            print("Unable to transform input to output because of bad indexing")
        try:
            inp_dset[counter1:counter1+INPUT_SIZE] = im_ip_wo_alpha.flatten()
            op_dset[counter1:counter1+INPUT_SIZE] = im_op_wo_alpha.flatten()
            counter1 = counter1+INPUT_SIZE
        except:
            print("Unable to Write out the inputs to the file handle")
            pdb.set_trace()
        try:
            transf_dset[counter2:counter2+3] = np.array([xx_diff,yy_diff,zz_diff]).flatten()
            counter2 = counter2 + 3
        except:
            print("Unable to write this tuple of transformations into the dataset")
    h5_input.flush()
    h5_output.flush()
    h5_transf.flush()
    h5_input.close()
    h5_output.close()
    h5_transf.close()
    return 1


def move_train_test(src=None,ignore=None):
    if src is None:
        src='/media/mudigonda/Gondor/Data/3dFace/rotatedFaces/'
    if ignore is None:
        ignore=['MACF312.mat','test','train','bad']
    #Find the list of files from src
    files,face_id,xx,yy,zz = make_fname_lists(src)

    #Find unique Face ID
    unique_items=[]
    for item in face_id:
        if item not in unique_items:
           unique_items.append(item)
    
    #Take 75% of them and put them in train, verify ignore
    for ii in range(150):
        if unique_items[ii] not in ignore:
            print('Moving File',unique_items[ii])
            mv_files = glob.glob(src+unique_items[ii]+'*')
            for file_to_move in mv_files:
                shutil.move(file_to_move,src+'/train')
    #Take the remaining and put them in test, verify ignore
    for ii in range(150,len(unique_items)): 
        if unique_items[ii] not in ignore:
            print('Moving File',unique_items[ii])
            mv_files = glob.glob(src+unique_items[ii]+'*')
            for file_to_move in mv_files:
                shutil.move(file_to_move,src+'/test')


    return 1


def crop_images(path):
    os.chdir(path)
    files_to_crop=glob.glob("*")
    for ii in range(len(files_to_crop)):
        if np.mod(ii,1000)==0:
            print('The value of the counter is ',ii)
        try:
            call('convert -crop 200X200+25+0 '+files_to_crop[ii]+' '+files_to_crop[ii],shell=True)
        except:
            print('could not convert')
    os.chdir('/media/mudigonda/Gondor/Projects/shape-basis-3d/python')
    return 1

def resize_images(path,resize_params=None):
    os.chdir(path)
    if resize_params is None:
        resize_params = '64x64'
    files_to_resize=glob.glob("*")
    for ii in range(len(files_to_resize)):
        if np.mod(ii,1000)==0:
            print('The value of the counter is ',ii)
        try:
            call('convert -resize '+resize_params+' '+files_to_resize[ii]+' resized/'+files_to_resize[ii],shell=True)
        except:
            print('could not resize')
    os.chdir('/media/mudigonda/Gondor/Projects/shape-basis-3d/python')
    return 1

def read_mean_txt(fileName):
    with open(fileName,'r') as f:
        l = f.readlines()
        mn = [float(i.split()[0]) for i in l]
        mn = np.array(mn)
        mn = mn/np.max(mn)
        mn = np.reshape(mn,[64,64,3])
    return mn
