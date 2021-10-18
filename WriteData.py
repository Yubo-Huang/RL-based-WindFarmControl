import numpy as np
import subprocess

'''
This file is used to write the weights of MLP to a file which will be tranmitted to OpenFAST to control the torque of each turbine.
The structure of this file:
TITLE
Loc_x, Loc_y, Loc_z
n = [n1, n2]    # number of neural units in the 1st and 2nd layer
w1              # weights of the 1st layer
b1              # bias of the 1st layer
w2              # weights of the 2nd layer
b2              # bias of the 2nd layer
w3              # weights of the 3rd layer
b3              # bias of the 3rd layer
'''

class Write_MLP(object):
    def __init__(self, fn, w):
        '''
        Str : fn, the file name for writing the weights of MLP
        List: w,  generated from MLP by tensorflow.
        '''
        
        self.fn = fn
        self.title = "The weights of three layers MLP"
        self.n   = np.array([32, 8])
        self.n   = self.n.reshape((1, 2))
        self.w1  = w[0].T
        self.b1  = w[1]
        self.b1  = self.b1.reshape((self.b1.size, 1))
        self.w2  = w[2].T
        self.b2  = w[3]
        self.b2  = self.b2.reshape((self.b2.size, 1))
        self.w3  = w[4].T
        self.b3  = w[5]
        self.b3  = self.b3.reshape((self.b3.size, 1))
    
    def Write_File(self):
        command = "rm %s" % self.fn
        subprocess.call(command, shell=True)
        command = "touch %s" % self.fn
        subprocess.call(command, shell=True)

        with open(self.fn, "w") as f:
            f.writelines(self.title)
            f.writelines("\n")
            np.savetxt(f, self.n, fmt='%d')
            np.savetxt(f, self.w1, fmt='%.06f')
            np.savetxt(f, self.b1, fmt='%.06f')
            np.savetxt(f, self.w2, fmt='%.06f')
            np.savetxt(f, self.b2, fmt='%.06f')
            np.savetxt(f, self.w3, fmt='%.06f')
            np.savetxt(f, self.b3, fmt='%.06f')