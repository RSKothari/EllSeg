import os

import cv2
import sys
import glob
import argparse
import matplotlib
import numpy as np
import deepdish as dd
import scipy.io as scio

sys.path.append('..')
from helperfunctions import mypause, generateEmptyStorage

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds',
                    help='Path to dataset',
                    type=str,
                    default='/media/rakshit/Monster/Datasets')
args = parser.parse_args()
if args.noDisp:
    noDisp = True
    print('No graphics')
else:
    noDisp = False
    print('Showing figures')

gui_env = ['Qt5Agg','WXAgg','TKAgg','GTKAgg']
for gui in gui_env:
    try:
        print("testing: {}".format(gui))
        
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue

print("Using: {}".format(matplotlib.get_backend()))
plt.ion()

PATH_DIR = os.path.join(args.path2ds, 'LPW')
PATH_DS = os.path.join(args.path2ds, 'All')
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')
list_ds = list(os.walk(PATH_DIR))[0][1]

print('Extracting LPW')

Image_counter = 0.0
ds_num = 0

def readFormattedText(path2file, ignoreLines):
    data = []
    count = 0
    f = open(path2file, 'r')
    for line in f:
        d = [float(d) for d in line.split()]
        count = count + 1
        if d and count > ignoreLines:
            data.append(d)
    f.close()
    return data

for name in list_ds:
    # Ignore the first row and column.
    # Columns: [index, p_x, p_y]
    opts = glob.glob(os.path.join(PATH_DIR, name, '*.avi'))
    for Path2vid in opts:
        # Read pupil  info
        loc, fName = os.path.split(Path2vid)
        fName = os.path.splitext(fName)[0]
        Path2text = os.path.join(loc, fName+'.txt')
        PupilData = np.array(readFormattedText(Path2text, 0))
        VidObj = cv2.VideoCapture(Path2vid)

        ds_name = ds_name = 'LPW_{}_{}_{}'.format(name, fName, ds_num)

        # Generate empty dictionaries
        Data, keydict = generateEmptyStorage(name='LPW', subset='LPW_{}'.format(name))
        if not noDisp:
            fig, plts = plt.subplots(1,1)
        fr_num = 0
        while(VidObj.isOpened()):
            ret, I = VidObj.read()
            if ret == True:
                I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                Data['Images'].append(I)
                keydict['resolution'].append(I.shape)
                keydict['archive'].append(ds_name)
                pupil_loc = PupilData[fr_num, :]
                keydict['pupil_loc'].append(pupil_loc)
                Data['pupil_loc'].append(pupil_loc)
                Data['Info'].append(str(fr_num))
                fr_num+=1
                Image_counter+=1
                if not noDisp:
                    if fr_num == 1:
                        cI = plts.imshow(I, cmap='gray')
                        cX = plts.scatter(pupil_loc[0], pupil_loc[1])
                        plt.show()
                        plt.pause(.01)
                    else:
                        newLoc = np.array([pupil_loc[0], pupil_loc[1]])
                        cI.set_data(I)
                        cX.set_offsets(newLoc)
                        mypause(0.01)
            else: # No more frames to load
                break
        Data['Images'] = np.stack(Data['Images'], axis=0)
        Data['pupil_loc'] = np.stack(Data['pupil_loc'], axis=0)
        keydict['resolution'] = np.stack(keydict['resolution'], axis=0)
        keydict['archive'] = np.stack(keydict['archive'], axis=0)
        keydict['pupil_loc'] = np.stack(keydict['pupil_loc'], axis=0)
        # Save data
        dd.io.save(os.path.join(PATH_DS, str(ds_name)+'.h5'), Data)
        scio.savemat(os.path.join(PATH_MASTER, str(ds_name)+'.mat'), keydict, appendmat=True)
        ds_num=ds_num+1
