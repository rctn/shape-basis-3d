import numpy as np
import pdb
#Testing out things about projection matrices

def make_im(X,Y,Z,bin_size):

    #Find the smallest Z and subtract it
    min_Z = Z.min()
    Z = Z - min_Z + 5.0
    
    scale = 10.0

    x = scale*X / Z
    y = scale*Y / Z

    #Creating bins

    min_x, min_y = np.min(x), np.min(y)
    max_x, max_y = np.max(x), np.max(y)
    mn_bin = np.min((min_x,min_y))
    mx_bin = np.max((max_x,max_y))
    print('Min and Max Value of bins ', mn_bin, mx_bin)

    bin_entries = np.linspace(mn_bin,mx_bin,bin_size)

    im = np.zeros((bin_size,bin_size))

    for ii in np.arange(bin_size):
        for jj in np.arange(bin_size):
            x_bin_val = np.where(bin_entries>=X[ii,jj])[0][0]
            y_bin_val = np.where(bin_entries>=Y[ii,jj])[0][0]

            im[x_bin_val,y_bin_val] = np.max((im[x_bin_val,y_bin_val],Z[ii,jj]))
            #im[x_bin_val,y_bin_val] = Z[x_bin_val,y_bin_val]
            #im[x_bin_val,y_bin_val] = np.max((im[x_bin_val,y_bin_val],Z[x_bin_val,y_bin_val]))
            #im[x_bin_val,y_bin_val] = Z[ii,jj]
            #im[ii,jj] = Z[ii,jj]


    return im
