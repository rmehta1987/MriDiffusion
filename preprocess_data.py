"""
This file preprocesses the data given by two folders.  The first folder
contains the Beta, alpha, and B paramater maps.  The second folder contains the
ROI of a slice from the corresponding map.  There is one ROI among among a number of
slices in the ROI folder.  We first find this ROI, and then extract it
element-wise multiplication to the corresponding map.  

--- model_results folder contents is identified by PAT****_TumorROI.mat
where **** is the patient number starting from 0001, the .mat files 
contain 6 files:
mlf_alpha
mlf_beta
mlf_conv
mlf_DDC
mlf_error
mlf_resids
mlf_SO

--- ROIS folder contents is identified by PAT****_TumorROI.mat
where **** is the patient number starting from 0001

This file then creates a data folder that matches the dataloader format 
for training in pytorch

"""

import scipy.io as sio  # loads matlab files
import glob  # loads filenames in the folders
import re  # import regex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from combat import neuroCombat


#This function is hard-coded, as we want to to know the individual maps extracted from the Mat Files
def createFiles(foldernames):
    '''@foldernames is the name of the folder where maps will be saved and should be 
        sorted as [amaps, bmaps, ddc-maps, diffmaps, perfmaps, fmaps]'''
    '''Directors are in current directory, model_results and ROI, 
    finds corresponding slices and then creates a file 
    with the ROI parameter map and the label [ [parameter map] [label]] '''

    # get files and sort them by name
    adict = {}
    bdict = {}
    ddict = {}
    idiff_dict = {}
    iperf_dict = {}
    if_dict = {}
    modelFiles = sorted(glob.glob('model_results/*.mat')) #A, B, DDC maps
    ivimFiles = sorted(glob.glob('ivim/*ivim*.mat')) #IVIM Files 
    rois = sorted(glob.glob('ROIS/*.mat'))
    csvlabels = pd.read_csv('FROC_Patient_Diagnosis.csv')
    #patientset = set() #store patient names, since patients may have multiple slices
    for i in range(0,len(modelFiles)):
        matmodel = sio.loadmat(modelFiles[i])
        ivimmodel = sio.loadmat(ivimFiles[i])
        matIdiff = ivimmodel['ivim_Ddiff']
        matIperf = ivimmodel['ivim_Dperf']
        matIf = ivimmodel['ivim_f']
        matalpha = matmodel['mlf_alpha']
        matbeta = matmodel['mlf_beta']
        matddc = matmodel['mlf_DDC']
        matroi = sio.loadmat(rois[i])['ROI']
        # make sure patient file names are same
        regPatient = re.compile(r"(?=P)(.*?)(?=_)")
        patientname = regPatient.search(modelFiles[i]).group()
        patientname2 = regPatient.search(ivimFiles[i]).group()
        if patientname in rois[i] and patientname2 in rois[i]:

            indexRoi = np.transpose(np.nonzero(matroi)) # get get index of slice where ROI exists
            
            # check if multiple ROIS
            uniqueInd = np.unique(indexRoi[:,2])
             #Assumes all ROIs correspond to one label (hard-coded for patient 15) which is a benign patient
            for j in range(0,len(uniqueInd)):
                #need to make sure all ROI masks are binary 0/1
                tempROI = matroi[:,:,uniqueInd[j]]
                tempROI = (tempROI > 0.5).astype(int)

                if j == 1 and i == 14:
                    mask_alpha = np.multiply(matalpha[:,:,uniqueInd[j]],tempROI)
                    mask_beta = np.multiply(matbeta[:,:,uniqueInd[j]],tempROI)
                    mask_ddc = np.multiply(matddc[:,:,uniqueInd[j]],tempROI)
                    mask_idiff = np.multiply(matIdiff[:,:,uniqueInd[j]],tempROI)
                    mask_iperf = np.multiply(matIperf[:,:,uniqueInd[j]],tempROI)
                    mask_if = np.multiply(matIf[:,:,uniqueInd[j]],tempROI)
                    #np.save('masks/unnorm_mask_alpha_%s_%d'%(patientname,j),[mask_alpha, 0])
                    #np.save('masks/unnorm_mask_beta_%s_%d'%(patientname,j),[mask_beta, 0])
                    #np.save('masks/unnorm_mask_ddc_%s_%d'%(patientname,j),[mask_ddc, 0])
                    adict['%s_%s_%d'%(foldernames[0],patientname,j)]=[mask_alpha, 0, patientname]
                    bdict['%s_%s_%d'%(foldernames[1],patientname,j)]=[mask_beta, 0, patientname]
                    ddict['%s_%s_%d'%(foldernames[2],patientname,j)]=[mask_ddc, 0, patientname]
                    idiff_dict['%s_%s_%d'%(foldernames[3],patientname,j)] = [mask_idiff, 0, patientname]
                    iperf_dict['%s_%s_%d'%(foldernames[4],patientname,j)] = [mask_iperf, 0, patientname]
                    if_dict['%s_%s_%d'%(foldernames[5],patientname,j)] = [mask_if, 0, patientname]
                    #patientset.add(patientname)
                else:
                    mask_alpha = np.multiply(matalpha[:,:,uniqueInd[j]],tempROI)
                    mask_beta = np.multiply(matbeta[:,:,uniqueInd[j]],tempROI)
                    mask_ddc = np.multiply(matddc[:,:,uniqueInd[j]],tempROI)
                    mask_idiff = np.multiply(matIdiff[:,:,uniqueInd[j]],tempROI)
                    mask_iperf = np.multiply(matIperf[:,:,uniqueInd[j]],tempROI)
                    mask_if = np.multiply(matIf[:,:,uniqueInd[j]],tempROI)
                    #np.save('masks/unnorm_mask_alpha_%s_%d'%(patientname,j),[mask_alpha, csvlabels['biopsy_binary'][i]])
                    #np.save('masks/unnorm_mask_beta_%s_%d'%(patientname,j),[mask_beta, csvlabels['biopsy_binary'][i]])
                    #np.save('masks/unnorm_mask_ddc_%s_%d'%(patientname,j),[mask_ddc, csvlabels['biopsy_binary'][i]])
                    adict['%s_%s_%d'%(foldernames[0],patientname,j)]=[mask_alpha, csvlabels['biopsy_binary'][i], patientname]
                    bdict['%s_%s_%d'%(foldernames[1],patientname,j)]=[mask_beta, csvlabels['biopsy_binary'][i], patientname]
                    ddict['%s_%s_%d'%(foldernames[2],patientname,j)]=[mask_ddc, csvlabels['biopsy_binary'][i], patientname]
                    idiff_dict['%s_%s_%d'%(foldernames[3],patientname,j)] = [mask_idiff, csvlabels['biopsy_binary'][i], patientname]
                    iperf_dict['%s_%s_%d'%(foldernames[4],patientname,j)] = [mask_iperf, csvlabels['biopsy_binary'][i], patientname]
                    if_dict['%s_%s_%d'%(foldernames[5],patientname,j)] = [mask_if, csvlabels['biopsy_binary'][i], patientname]
                    #patientset.add(patientname)

    return adict, bdict, ddict, idiff_dict, iperf_dict, if_dict

def pad_image(img, pad_t, pad_r, pad_b, pad_l):
    """Add padding of zeroes to an image.
    Add padding to an array image.
    :param img:
    :param pad_t:
    :param pad_r:
    :param pad_b:
    :param pad_l:
    """
    height, width = img.shape

    # Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    # Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    # Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    # Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img

def center_image(img, cropped_image):
    """Return a centered image.
    :param img: original image
    :param cropped_image: cropped image:
    """
    zero_axis_fill = (img.shape[0] - cropped_image.shape[0])
    one_axis_fill = (img.shape[1] - cropped_image.shape[1])

    top = int(zero_axis_fill / 2)
    bottom = int(zero_axis_fill - top)
    left = int(one_axis_fill / 2)
    right = int(one_axis_fill - left)

    padded_image = pad_image(cropped_image, top, left, bottom, right)

    return padded_image

def padFiles(alldicts, name):
    '''This changes the size of the parameter maps to a square matrix of size 64 x 64.  Everything else
    is zero-padded'''

    #Iterate over every file in a dict (folder of the map)
    for j, thedict in enumerate(alldicts):
        thekeys = thedict.keys()
        for i, themap in enumerate(thekeys):
            onemap = thedict[themap][0]
            pixels = np.transpose(np.nonzero(onemap))
            max_x_p = np.max(pixels[:,0])
            min_x_p = np.min(pixels[:,0])
            max_y_p = np.max(pixels[:,1])
            min_y_p = np.min(pixels[:,1])
            mapcrop = onemap[min_x_p:max_x_p,min_y_p:max_y_p]
            mappad = center_image(np.zeros((64,64)), mapcrop)
            mappad = mappad.reshape(64,64)
            thedict[themap][0] = mappad
        alldicts[j] = thedict

    return alldicts   

def padFilesasList(allFiles):
    '''This changes the size of the parameter maps to a square matrix of size 64 x 64.  Everything else
    is zero-padded'''

    #Iterate over every file in a dict (folder of the map)
    for j, thefiles in enumerate(allFiles):
        for i, themap in enumerate(thefiles):
            onemap = themap[0]
            pixels = np.transpose(np.nonzero(onemap))
            max_x_p = np.max(pixels[:,0])
            min_x_p = np.min(pixels[:,0])
            max_y_p = np.max(pixels[:,1])
            min_y_p = np.min(pixels[:,1])
            mapcrop = onemap[min_x_p:max_x_p,min_y_p:max_y_p]
            mappad = center_image(np.zeros((64,64)), mapcrop)
            mappad = mappad.reshape(64,64)
            allFiles[j][i][0] = mappad
        

    return allFiles

def MaxMinNorm(afiles, bfiles, dfiles):
    '''Use min-max standardization'''

    #vectorize each map and then calculate mean and standard deviation
    amaps = np.array([temp.reshape(-1) for temp in np.array(afiles)[:,0]])
    amax = np.max(amaps.ravel()[np.flatnonzero(amaps)])
    if amax < 1:
        newamax = 1
    else:
        newamax = amax + amin
    amin = np.min(amaps.ravel()[np.flatnonzero(amaps)])
 

    bmaps = np.array([temp.reshape(-1) for temp in np.array(bfiles)[:,0]])
    bmax = np.max(bmaps.ravel()[np.flatnonzero(bmaps)])
    if bmax < 1:
        newbmax = 1
    else:
        newbmax = bmax + bmin
    bmin = np.min(bmaps.ravel()[np.flatnonzero(bmaps)])

    dmaps = np.array([temp.reshape(-1) for temp in np.array(dfiles)[:,0]]) 
    dmax = np.max(dmaps.ravel()[np.flatnonzero(dmaps)])
    if dmax < 1:
        newdmax = 1
    else:
        newdmax = dmin + dmax
    dmin = np.min(dmaps.ravel()[np.flatnonzero(dmaps)])

    #apply normalization to each map:
    for i,(amap,bmap,dmap) in enumerate(zip(amaps,bmaps,dmaps)):
        
        amap[amap!=0] = ((amap[amap!=0] - amin)/(amax - amin))*newamax + amin
        amaps[i] = amap
        bmap[bmap!=0]= ((bmap[bmap!=0] - bmin)/(bmax - bmin))*newbmax + bmin
        bmaps[i] = bmap
        dmap[dmap!=0] = ((dmap[dmap!=0] - dmin)/(dmax - dmin))*newdmax + dmin
        dmaps[i] = dmap

    #replace original maps with normalized maps

    for i, _ in enumerate(afiles):
        afiles[i][0] = amaps[i].reshape(64,64)
        bfiles[i][0] = bmaps[i].reshape(64,64)
        dfiles[i][0] = dmaps[i].reshape(64,64)

    return afiles, bfiles, dfiles

#Currently Does not work, most likely not needed
def combatFiles(afiles, bfiles, dfiles):
    '''Use combat normalization to normalize instead of quantile sample'''
    '''Need to create a datafram that matches the Combat Script input style
        
    data : a pandas data frame or numpy array
    neuroimaging data to correct with shape = (samples, features)
    e.g. cortical thickness measurements, image voxels, etc

    covars : a pandas data frame w/ shape = (samples, features)
        demographic/phenotypic/behavioral/batch data 
        
    batch_col : string
        - batch effect variable
        - e.g. scan site
    
    Need to create a normalization for each parameter map
    
    --- Need to figure out how to normalize on Volume and not batch
    
    '''

    #First put each parameter map into to panda dataframe, with columns as 
    #patient_name and rows as the features

    #numpy array for parameters and the voxel values
    amaps = np.array([temp.reshape(-1) for temp in np.array(afiles)[:,0]])
    bmaps = np.array([temp.reshape(-1) for temp in np.array(bfiles)[:,0]])
    dmaps = np.array([temp.reshape(-1) for temp in np.array(dfiles)[:,0]]) 
    batchname = [temp for temp in np.array(afiles)[:,2]]
    batchFrame = pd.DataFrame({'batch': batchname})

    tamaps = neuroCombat(amaps,batchFrame, 'batch')
    tbmaps = neuroCombat(bmaps,batchFrame, 'batch')
    tdmaps = neuroCombat(dmaps,batchFrame, 'batch')
    

def zscoreNorm(afiles, bfiles, dfiles):
    ''' Normalize the respective maps by the z-score:  [v_i - mean(V)] / std(V) where v_i is voxel and V are all the voxels'''

    #vectorize each map and then calculate mean and standard deviation
    amaps = np.array([temp.reshape(-1) for temp in np.array(afiles)[:,0]])
    amean = np.true_divide(amaps.sum(),(amaps!=0).sum())
    astd = (amaps!=0).std()

    bmaps = np.array([temp.reshape(-1) for temp in np.array(bfiles)[:,0]])
    bmean = np.true_divide(bmaps.sum(),(bmaps!=0).sum())
    bstd = (bmaps!=0).std()

    dmaps = np.array([temp.reshape(-1) for temp in np.array(dfiles)[:,0]]) 
    dmean = np.true_divide(dmaps.sum(),(dmaps!=0).sum())
    dstd = (dmaps!=0).std()

    #apply normalization to each map:

    amaps[amaps!=0] = (amaps[amaps!=0] - amean)/astd
    bmaps[bmaps!=0] = (bmaps[bmaps!=0] - bmean)/bstd
    dmaps[dmaps!=0] = (dmaps[dmaps!=0] - dmean)/dstd
    #replace original maps with normalized maps

    for i, _ in enumerate(afiles):
        afiles[i][0] = amaps[i].reshape(64,64)
        bfiles[i][0] = bmaps[i].reshape(64,64)
        dfiles[i][0] = dmaps[i].reshape(64,64)

    return afiles, bfiles, dfiles

#This function is slightly hard-coded as we want a list to identify individual maps, in this case we are assuming 6 different diffusion maps
def loadexisting(file_path, name):
    '''@file_path is where the files are located
        @name is the names of the individual files, in this case we are assuming names are sorted as [amaps, bmaps, ddc-maps, diffmaps, perfmaps, fmaps]'''
    afiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[0])))
    bfiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[1])))
    dfiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[2])))
    difffiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[3])))
    perffiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[4])))
    ffiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[5])))

    lafiles = []
    lbfiles = []
    ldfiles = []
    ldiff_files = []
    lperf_files = []
    lf_files = []

    for i, (a,b,d,e,f,g) in enumerate(zip(afiles,bfiles,dfiles,difffiles,perffiles,ffiles)):
        tmap, label, patientname = np.load(a)
        lafiles.append([tmap, label, patientname])
        tmap = np.load(b)[0]
        lbfiles.append([tmap, label, patientname])
        tmap = np.load(d)[0]
        ldfiles.append([tmap, label, patientname])
        tmap = np.load(e)[0]
        ldiff_files.append([tmap, label, patientname])
        tmap = np.load(f)[0]
        lperf_files.append([tmap, label, patientname])
        tmap = np.load(g)[0]
        lf_files.append([tmap, label, patientname])


    return lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files

def savewithPatient(file_path, name, allfiles):
        
    for  i, thefiles in enumerate(allfiles):
        for j, onefile in enumerate(thefiles):
            patientname = onefile[2]
            label = onefile[1]
            af = file_path + '/%s_%d_%s'%(patientname,j,name[i])
            np.save(af,[onefile[0], label, patientname])

def saveDicts(alldicts, name):
    
    for i,thedict in enumerate(alldicts):
        thekeys = thedict.keys()
        for akey in thekeys:
            filename = akey+ '_' + name[i]
            np.save(filename,[thedict[akey][0],thedict[akey][1],thedict[akey][2]])


def runProcess():

    #-----Create Files from Masks and Original MRI slices
    
    filenames = ['mmasks/mask_alpha_','mmasks/mask_beta_','mmasks/mask_ddc_','mmasks/mask_diff_','mmasks/mask_perf_','mmasks/mask_f_']
    adict, bdict, ddict, diff_dict, perf_dict, f_dict = createFiles(filenames)
    print ("created Files")

    #saved Masked Files
    name = ['amask', 'bmask', 'dmask', 'diffmask', 'perfmask', 'fmask']
    saveDicts([adict, bdict, ddict, diff_dict, perf_dict, f_dict], name)
    print ("saved masked Files")

    #Pad Files to 64x64 - if running first time
    '''alldicts = [adict, bdict, ddict, diff_dict, perf_dict, f_dict]
    name = ['apad', 'bpad', 'dpad', 'diffpad', 'perfpad', 'fpad']
    alldicts = padFiles(alldicts, name)
    print ("padded files")

    #Save Padded Files
    saveDicts(alldicts, name)
    print ("saved padded Files")'''

    #Load Masked Files and Pad
    file_path = 'mmasks'
    name = ['amask', 'bmask', 'dmask', 'diffmask', 'perfmask', 'fmask']
    lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files = loadexisting(file_path, name)
    
    
    print ("Number of files extracted (should be 40) from loading masks: %d"%(len(lafiles)))
    
    lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files = padFilesasList([lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files])
    print ("padded files")

    name = ['apad', 'bpad', 'dpad', 'diffpad', 'perfpad', 'fpad']
    savewithPatient(file_path, name, [lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files])
    print ("saved padded Files")

    #load padded images to a list, useful for when files already exist;
    #file_path = 'mmasks'
    #lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files = loadexisting(file_path, name)
    #print ("loaded existing files")

    #Use Max-Min normalization:
    lafiles, lbfiles, ldfiles = MaxMinNorm(lafiles, lbfiles, ldfiles)
    ldiff_files, lperf_files, lf_files = MaxMinNorm(ldiff_files, lperf_files, lf_files)

    #save the normalized maps
    print ("normalizing Maps using Max-Min")
    file_path = 'maxmin'
    name = ['mm_apad', 'mm_bpad', 'mm_dpad', 'mm_diffpad', 'mm_perfpad', 'mm_fpad']
    savewithPatient(file_path, name, [lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files])
    print("Finished saving files")

'''
#load files, names of the padded files:
name = ['apad', 'bpad', 'dpad', 'diffpad', 'perfpad', 'fpad']
#where are files located:
filepath = 'mmasks'

#load files:
lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files = loadexisting(filepath, name) 
#normalize the images using min-max
lafiles, lbfiles, ldfiles = MaxMinNorm(lafiles, lbfiles, ldfiles)
ldiff_files, lperf_files, lf_files = MaxMinNorm(ldiff_files, lperf_files, lf_files)

#save the normalized files, create a name and set folder path, folder must already be created
filepath = 'maxmin'
name = ['mm_apad', 'mm_bpad', 'mm_dpad', 'mm_diffpad', 'mm_perfpad', 'mm_fpad']
savewithPatient(filepath, name, [lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files])
'''