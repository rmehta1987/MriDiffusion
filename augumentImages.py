import glob  # loads filenames in the folders
import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt



def getFiles(file_path,name):

    afiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[0])))
    bfiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[1])))
    dfiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[2])))
    lafiles = []
    lbfiles = []
    ldfiles = []

    print ("obtaining files")
    for i, (a,b,d) in enumerate(zip(afiles, bfiles, dfiles)):
        lafiles.append(np.load(a))
        lbfiles.append(np.load(b))
        ldfiles.append(np.load(d))

    
    return lafiles, lbfiles, ldfiles

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

def createSmallerCrops(img):
#creates small images, but finding the ROI and then dividing it by 3
#image needs to be centered, and are then re-centered again after division
    
    timg = np.ravel(img)
    tind = np.nonzero(timg)[0]

    #split indices into 3 piecies
    firstsplit = int(len(tind)/3)
    secondsplit = firstsplit*2

    timg1 = np.zeros((len(timg)))
    timg2 = np.zeros((len(timg)))
    timg3 = np.zeros((len(timg)))

    timg1[tind[0:firstsplit]] = timg[tind[0:firstsplit]]
    timg2[tind[firstsplit:secondsplit+1]] = timg[tind[firstsplit:secondsplit+1]]
    timg3[tind[secondsplit+1:]] = timg[tind[secondsplit+1:]]

    #recenter and pad images
    timg1 = np.reshape(timg1,(64,64))
    timg2 = np.reshape(timg2,(64,64))
    timg3 = np.reshape(timg3,(64,64))

    return center_image(np.zeros((64,64)), timg1), center_image(np.zeros((64,64)), timg2), center_image(np.zeros((64,64)), timg3)


def saveFiles(file_path, name, onames, afiles, amap, bmap, dmap):
    
    for  i, (afile,amap,bmap,dmap) in enumerate(zip(afiles,amap,bmap,dmap)):
        
        patientname = afile[2]
        label = afile[1]
        af = file_path + '/%s_%s_%s_%d'%(name,onames[0],patientname,i)
        np.save(af,[amap, label, patientname])
        af = file_path + '/%s_%s_%s_%d'%(name,onames[1],patientname,i)
        np.save(af,[bmap, label, patientname])
        af = file_path + '/%s_%s_%s_%d'%(name,onames[2],patientname,i)
        np.save(af,[dmap, label, patientname])


def augImages(filepath, fnames, onames, ofilepath):
    
    
    afiles, bfiles, dfiles = getFiles(filepath, fnames)

    samap1 = []
    samap2 = []
    samap3 = []

    sbmap1 = []
    sbmap2 = []
    sbmap3  = []

    sdmap1 = []
    sdmap2 = []
    sdmap3 = []

    #create crops of original maps, splits into 3 crops

    for i, (amap,bmap,dmap) in enumerate(zip(afiles,bfiles,dfiles)):

        temp1, temp2, temp3 = createSmallerCrops(amap[0])
        samap1.append(temp1)
        samap2.append(temp2)
        samap3.append(temp3)
        temp1, temp2, temp3 = createSmallerCrops(bmap[0])
        sbmap1.append(temp1)
        sbmap2.append(temp2)
        sbmap3.append(temp3)
        temp1, temp2, temp3 = createSmallerCrops(dmap[0])
        sdmap1.append(temp1)
        sdmap2.append(temp2)
        sdmap3.append(temp3)


    #Now do random horizontal flips, rotations for all images

    #augmentations of original data
    augamap = []
    augbmap = []
    augdmap = []

    #augmentations of cropped images
    caugamap1 = []
    cbugamap1 = []
    cdugamap1 = []

    caugamap2 = []
    cbugamap2 = []
    cdugamap2 = []

    caugamap3 = []
    cbugamap3 = []
    cdugamap3 = []





    for i, (amap,bmap,dmap,camap1,camap2,camap3,cbmap1,cbmap2,cbmap3, cdmap1,cdmap2,cdmap3) in enumerate(zip(afiles,bfiles,dfiles,samap1, samap2, samap3, sbmap1, sbmap2, sbmap3, sdmap1, sdmap2, sdmap3)):
        
        
        #flip image vertically or horizontally - bernoulli
        if np.random.randint(0,2):
            # Now do a random rotation
            randrot = np.random.randint(30,331) #rotate between 30 to 331 degrees

            augamap.append(rotate(np.fliplr(amap[0]),randrot))
            augbmap.append(rotate(np.fliplr(bmap[0]),randrot))
            augdmap.append(rotate(np.fliplr(dmap[0]),randrot))
            caugamap1.append(rotate(np.fliplr(camap1),randrot))
            cbugamap1.append(rotate(np.fliplr(cbmap1),randrot))
            cdugamap1.append(rotate(np.fliplr(cdmap1),randrot))
            caugamap2.append(rotate(np.fliplr(camap2),randrot))
            cbugamap2.append(rotate(np.fliplr(cbmap2),randrot))
            cdugamap2.append(rotate(np.fliplr(cdmap2),randrot))
            caugamap3.append(rotate(np.fliplr(camap3),randrot))
            cbugamap3.append(rotate(np.fliplr(cbmap3),randrot))
            cdugamap3.append(rotate(np.fliplr(cdmap3),randrot))
        else:

            # Now do a random rotation
            randrot = np.random.randint(30,331) #rotate between 30 to 331 degrees

            augamap.append(rotate(np.flipud(amap[0]),randrot))
            augbmap.append(rotate(np.flipud(bmap[0]),randrot))
            augdmap.append(rotate(np.flipud(dmap[0]),randrot))
            caugamap1.append(rotate(np.flipud(camap1),randrot))
            cbugamap1.append(rotate(np.flipud(cbmap1),randrot))
            cdugamap1.append(rotate(np.flipud(cdmap1),randrot))
            caugamap2.append(rotate(np.flipud(camap2),randrot))
            cbugamap2.append(rotate(np.flipud(cbmap2),randrot))
            cdugamap2.append(rotate(np.flipud(cdmap2),randrot))
            caugamap3.append(rotate(np.flipud(camap3),randrot))
            cbugamap3.append(rotate(np.flipud(cbmap3),randrot))
            cdugamap3.append(rotate(np.flipud(cdmap3),randrot))

    # Save the augmented files

    print ("saving augmented files")

    file_path = ofilepath
    #OG Augmented Files
    name = 'ogaug'
    saveFiles(file_path, name, onames, afiles, augamap, augbmap, augdmap)

    #Crop Set 1 Maps Augmentations
    name = 'cropaug1'
    saveFiles(file_path, name, onames, afiles, caugamap1, cbugamap1, cdugamap1)

    #Crop Set 1 Maps Augmentations
    name = 'cropaug2'
    saveFiles(file_path, name, onames, afiles, caugamap2, cbugamap2, cdugamap2)

    #Crop Set 1 Maps Augmentations
    name = 'cropaug3'
    saveFiles(file_path, name, onames, afiles, caugamap3, cbugamap3, cdugamap3)




name = ['mm_apad', 'mm_bpad', 'mm_dpad']
filepath = 'maxmin'
onames = ['alpha', 'beta', 'ddc']
opath = 'newIvim_maxminAug'

augImages(filepath,name,onames,opath) 

'''
name = ['mm_diffpad', 'mm_perfpad', 'mm_fpad']
filepath = 'newIvim_maxmin'
onames = ['diff', 'perf', 'f']
opath = 'newIvim_maxminAug'

augImages(filepath,name,onames,opath) '''