#Shell Script that calls Pulkit's HDF5 to LevelDB files
#!/bin/bash
HDF52LDB=$HOME/Tools/caffe/build/tools/hdf52leveldb

#Train Input
echo "Converting Training Input"
$HDF52LDB /media/mudigonda/Gondor/Data/3dFace/train_input.hdf5 /media/mudigonda/Gondor/Data/3dFace/train_input 64 64 3 150000 Input

#Train Output
echo "Converting Training Output"
$HDF52LDB /media/mudigonda/Gondor/Data/3dFace/train_output.hdf5 /media/mudigonda/Gondor/Data/3dFace/train_output 64 64 3 150000 Output

#Train Transformations
echo "Converting Training Transformation"
$HDF52LDB /media/mudigonda/Gondor/Data/3dFace/train_transf.hdf5 /media/mudigonda/Gondor/Data/3dFace/train_transf 3 1 1 150000 Transformation

#Test Input
echo "Converting Testing Input"
$HDF52LDB /media/mudigonda/Gondor/Data/3dFace/test_input.hdf5 /media/mudigonda/Gondor/Data/3dFace/test_input 64 64 3 49000 Input

#Test Output
echo "Converting Testing Output"
$HDF52LDB /media/mudigonda/Gondor/Data/3dFace/test_output.hdf5 /media/mudigonda/Gondor/Data/3dFace/test_output 64 64 3 49000 Output

#Test Transformations
echo "Converting Testing Transformation"
$HDF52LDB /media/mudigonda/Gondor/Data/3dFace/test_transf.hdf5 /media/mudigonda/Gondor/Data/3dFace/test_transf 3 1 1 49000 Transformation


