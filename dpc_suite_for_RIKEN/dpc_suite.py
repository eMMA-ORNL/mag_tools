"""
This module contains all of the functions necessary for plotting the in-plane
magnetic inductance color maps from a dpc segmented detector input.
Assumes four pixels: A, B, C, D
"""
__author__ = "Fehmi Sami Yasin"
__date__ = "23/03/03"
__version__ = "1.1"
__maintainer__ = "Fehmi Sami Yasin"
__email__ = "yasinfs@ornl.gov"
__last_edit__ = "24/05/20"

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import smooth_hsv_jjc as smooth_hsv
import vector_tools as vt
from matplotlib.colors import colorConverter

def add_scalebar(data, pix, pix_unit, coord_system,
                 scalebar_fraction_of_image = 0.3,
                 loc = 'lower right', label_top = True,
                 sep = 5, frameon = False,
                 ftsize = 18, color = 'white'):
    '''
    Function that adds a scalebar to a plt.subplot image.
    '''
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    fontprops = fm.FontProperties(size=ftsize)
    print(np.shape(data), pix, pix_unit)
    SB_PIX_FRAC = (scalebar_fraction_of_image *
                   np.shape(data)[1]) # pix length of desired scalebar
    SB_UNITS = str(pix_unit)
    SB_NM = int(round(SB_PIX_FRAC * pix,-2)) # nm
    SB_PIX = SB_NM / pix # pix
    if np.logical_and(SB_PIX > SB_PIX_FRAC, SB_NM <=100) or SB_NM == 0:
        SB_NM = 50 #nm
        SB_PIX = SB_NM / pix # pix
    print('SB_NM = ', str(SB_NM))
    scalebar = AnchoredSizeBar(coord_system,
                               SB_PIX, str(str(SB_NM) + ' ' +
                                           str(SB_UNITS)),
                               str(loc), 
                               sep = sep,
                               label_top = label_top,
                               pad=1,
                               color = str(color),
                               frameon=frameon,
                               size_vertical=3,
                               fontproperties=fontprops)
    
    return(scalebar)

def calc_B_from_ABCD(A, B, C, D,
                     sv_file_name='', auto_remove_offset=False,
                     magnetic=True, normalize=True, save_field=False):
    '''
    Returns either the in-plane electric or magnetic field depending on the 'magnetic'
    keyword setting calculated from the DPC pixels A, B, C and D.
    '''

    # Define a box from which to calculate the DPC offset (center of mass of the beam)
    box_width = A.shape[0] // 4
    midy = A.shape[0] // 2
    midx = A.shape[1] // 2
    box = slice(midy - box_width // 2,
                midy + box_width // 2), slice(midx - box_width // 2,
                                              midx + box_width // 2)
    
    #Calculate pixel B-D and A-C
    BD = (np.copy(B)-np.copy(D))
    AC = (np.copy(A)-np.copy(C))#-np.load(FN_BX)

    if magnetic == False: #calculate electric field
        #Calculate the x component of the electric field from BD
        sx = np.copy(BD)
        if auto_remove_offset == True:
            offset_sx = (np.max(sx[box]) + np.min(sx[box]))/2
            sx = np.copy(sx) - offset_sx

        # Calculate sy from AC
        sy = np.copy(AC)
        if auto_remove_offset == True:
            offset_sy = (np.max(sy[box]) - np.abs(np.min(sy[box])))/2
            sy = np.copy(sy)-offset_sy

    if magnetic == True: #calculate the magnetic field
        #Calculate sy from BD
        sy = np.copy(BD)
        if auto_remove_offset == True:
            offset_sy = (np.max(sy[box]) + np.min(sy[box]))/2
            sy = np.copy(sy)-offset_sy

        # Calculate sx from AC
        sx = -np.copy(AC)
        if auto_remove_offset == True:
            offset_sx = (np.max(sx[box]) - np.abs(np.min(sx[box])))/2
            sx = np.copy(sx)-offset_sx
    # Normalize the field array
    s = np.zeros((3,1,np.shape(sx)[0],np.shape(sx)[1]))
    s[0] = np.copy(sx)
    s[1] = np.copy(sy)
    s[2] = np.zeros_like(sx)
    s_norm = (np.copy(s)/np.max(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)))
    #Saves data with shape (nz, ny, nx)
    if np.logical_and(sv_file_name!='', save_field==True):
        np.save(str(sv_file_name + '_sx'), s_norm[0])
        np.save(str(sv_file_name + '_sy'), s_norm[1])
    return(s_norm)

def calc_B_from_CoM(x_com, y_com,
                     sv_file_name='', auto_remove_offset=False,
                     magnetic=True, normalize=True, save_field=False):
    '''
    Returns either the in-plane electric or magnetic field depending on the 'magnetic'
    keyword setting calculated from the DPC pixels A, B, C and D.
    '''

    #Calculate pixel B-D and A-C
    BD = x_com
    AC = y_com

    if magnetic == False: #calculate electric field
        #Calculate the x component of the electric field from BD
        sx = np.copy(BD)

        # Calculate sy from AC
        sy = np.copy(AC)

    if magnetic == True: #calculate the magnetic field
        #Calculate sy from BD
        sy = np.copy(BD)

        # Calculate sx from AC
        sx = -np.copy(AC)
    # Normalize the field array
    s = np.zeros((3,1,np.shape(sx)[0],np.shape(sx)[1]))
    s[0] = np.copy(sx)
    s[1] = np.copy(sy)
    s[2] = np.zeros_like(sx)
    s_norm = (np.copy(s)/np.max(np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)))
    #Saves data with shape (nz, ny, nx)
    if np.logical_and(sv_file_name!='', save_field==True):
        np.save(str(sv_file_name + '_sx'), s_norm[0])
        np.save(str(sv_file_name + '_sy'), s_norm[1])
    return(s_norm)

def calc_dpc_offset(bx, by, box = False, box_width=10):
    '''Calculates the offset signal, i.e. center of mass of the disc.'''
  
    # Define a box from which to calculate the DPC offset (center of
    # mass of the beam)
    if box==False:
        box = slice(bx.shape[0] // 2 - box_width // 2,
                    bx.shape[0] // 2 + box_width // 2), slice(by.shape[0] // 2 -
                                                              box_width // 2,
                                                              by.shape[0] // 2 +
                                                              box_width // 2)    
    # Calculate offset from by
    #offset_by = (np.max(by[box]) + np.min(by[box]))/2
    offset_by = np.mean(by[box])

    # Calculate offset from bx
#    offset_bx = (np.max(bx[box]) + np.min(bx[box]))/2
    offset_bx = np.mean(bx[box])
    return(offset_bx, offset_by)

def DPC_colormap_from_ABCD(A, B, C, D, #sv_fmt='pdf',
                           sv_file_name='', auto_remove_offset=False,
                           im_w=10, arr_w=1/150, arr_scale=20, nm_pix=1.,
                           arrows=True, m_step=15, normalize=False,
                           intensity_scale=2,offset_angle=0,
                           scalebar = True, magnetic = True,
                           high_saturation=True, save_field=False,
                           vmin=False, vmax=False):
    '''
    Plot's the DPC colormap with or without arrows from an four pixel
    annulus detector with horizontal pixels A and C (relatable to B_y) and
    vertical pixels B and D (relatable to B_x).

    sv_fmt is the file format you want to save the image as.

    im_w is the field of view you'd like to save, im_w=0.5 crops the image in
    half from the center.

    nm_pix is the pixel length in nm/pix

    normalize=True will normalize the 2D array for you. Don't use if your data
    is already normalized.

    offset_angle rotates the in-plane vector by some user-defined angle in degrees.

    intensity_scale is the factor by which you want to amplify the intensity of
    the color in the image.

    scalebar = True plots a scalebar

    vmin and vmax set the contrast limits for the color plot
    '''
    PIX_NEW = nm_pix #nm/pix
    s_norm = calc_B_from_ABCD(A=A, B=B, C=C, D=D,
                              sv_file_name=sv_file_name,
                              auto_remove_offset=auto_remove_offset,
                              magnetic=magnetic, normalize=normalize,
                              save_field=save_field)
    
    plot_B_3D(b_arr = s_norm[:,0,::-1,:],#sv_fmt='jpg',
              sv_file_name=sv_file_name,
              im_w=im_w,arr_w=arr_w,arr_scale=arr_scale, nm_pix=nm_pix,
              arrows=arrows,
              m_step=m_step,normalize=normalize,intensity_scale=intensity_scale,
              offset_angle=offset_angle, scalebar = scalebar,
              high_saturation=high_saturation, vmin=vmin, vmax=vmax)

def plot_B_3D(b_arr, sv_file_name = '', sv_fmt = 'png', im_w = 1, nm_pix = 1,
              arr_w = .01, arr_scale = 25, m_step = 0, arrows = True,
              im_only = True, normalize=False, vmin = False, vmax = False,
              inset = False, inset_loc = [0,0,0.25,0.25],
              annote = '', annote_loc = [0.1, 0.1],
              intensity_scale = 1, offset_angle = 0, sv_rot_file_name = '',
              scalebar = False,
              intensity_norm=False, high_saturation=True):
    """
    Plotting a slice of a 3D vector field. B_ARR is the
    vector field array with shape [3, NY_PIX, NX_PIX].
    The first axis represents the x, y and z component of the field.
    NY_PIX and NX_PIX are the number of pixels in the y and x direction,
    respectively. They are usually equal, and its generally a good idea that
    they be factorable by 2^n.
    sv_file_name is the path to the save directory as well as file name:
    "path/to/directory/file_name"
    sv_fmt is the file format you'd like to save the images as
    im_w is the field of view you'd like to save, im_w=0.5 crops the image in
    half from the center.
    nm_pix is the pixel length in nm/pix
    normalize=True will normalize the 2D array for you. Don't use if your data
    is already normalized.
    offset_angle rotates the in-plane vector by some user-defined angle in degrees.
    """
    plt.clf()

    PIX=nm_pix #nm/pix
    pix_unit = 'nm'
    if offset_angle !=0:
        b_arr_temp = np.zeros((b_arr.shape[0],
                               1, b_arr.shape[1], b_arr.shape[2]))
        b_arr_temp[:,0,:,:] = np.copy(b_arr)
        b_arr_rot = rotate_3d_vec2(mag=np.copy(b_arr_temp),
                                   angx=[0],
                                   angy=[0],
                                   angz=[offset_angle],spins_only=True,
                                   pix=nm_pix,PAD=True,TEST=False)[0]
        b_arr = np.copy(np.array(b_arr_rot)[:,0,:,:])
        if sv_rot_file_name != '':
            np.save(sv_rot_file_name, b_arr) 
    MX = np.copy(b_arr[0])
    MY = np.copy(b_arr[1])
    b_dimensions = np.shape(b_arr)[0]
    #if b_dimensions==3:
    #    MZ = np.copy(b_arr[2])
    #else:
    MZ = np.ones_like(MX)
    MZ *= -np.max(np.sqrt(MX ** 2 + MY ** 2))
    MZ +=  np.sqrt(MX ** 2 + MY ** 2)
    MIDX = np.shape(b_arr)[2]//2
    MIDY = np.shape(b_arr)[1]//2
    NX_PIX = np.shape(b_arr)[2]
    NY_PIX = np.shape(b_arr)[1]
    IM_WIDTH_X = int(im_w*NX_PIX//2)
    IM_WIDTH_Y = int(im_w*NY_PIX//2)
    BOX = slice(MIDY-IM_WIDTH_Y,
                MIDY+IM_WIDTH_Y),slice(MIDX-IM_WIDTH_X,
                                       MIDX+IM_WIDTH_X)
    
    if normalize==True:
        norm_denom = np.max(np.sqrt(MX ** 2 + MY ** 2 + MZ **2))
        MX_NORM=(np.copy(MX) / norm_denom)[BOX]
        MY_NORM=(np.copy(MY) / norm_denom)[BOX]
        MZ_NORM=(np.copy(MZ) / norm_denom)[BOX]
    else:
        MX_NORM=np.copy(MX)[BOX]
        MY_NORM=np.copy(MY)[BOX]
        MZ_NORM=np.copy(MZ)[BOX]
    
    if vmin == False:
        vec_len = np.sqrt(MX ** 2 + MY ** 2 + MZ **2)
        vmin == np.min(vec_len)
    if vmax == False:
        vec_len = np.sqrt(MX ** 2 + MY ** 2 + MZ **2)
        vmax == np.max(vec_len)
    b_inplane = np.copy(MY_NORM) + 1.0j*np.copy(MX_NORM)
    #Set the arrow length to zero for in-plane vector amplitude<vmin
    b_inplane_amp = np.abs(b_inplane)
    b_inplane[b_inplane_amp<vmin] = 0
    rgba_b = smooth_hsv.smooth_rgba(MX_NORM, MY_NORM, MZ_NORM,
                                    high_saturation = high_saturation,
                                    min_cutoff = vmin,
                                    max_cutoff = vmax)

    FTSIZE=35
    TICKSIZE=0.75*FTSIZE
    FIG_SIZE_FACTOR = 7.5/np.max([np.shape(b_inplane)[0],np.shape(b_inplane)[1]])
    FIG_SIZE = (FIG_SIZE_FACTOR*np.shape(b_inplane)[1],
                FIG_SIZE_FACTOR*np.shape(b_inplane)[0])
 
    NUM_PTS = 5
    INCREMENT = int(round(PIX*np.shape(b_inplane)[0]//NUM_PTS,-1)) #nm
    if m_step==0:
        M_STEP=int(np.shape(MX_NORM)[0]//(2**5.5))
    else:
        M_STEP=int(m_step)
    print("M_STEP = "+str(M_STEP))
    TEMPX=np.copy(np.imag(b_inplane)[:-1:M_STEP,:-1:M_STEP])
    TEMPY=np.copy(np.real(b_inplane)[:-1:M_STEP,:-1:M_STEP])
    
    X1 = np.linspace(0, int(np.shape(b_inplane)[1]), np.shape(TEMPX)[1])
    Y1 = np.linspace(0, int(np.shape(b_inplane)[0]), np.shape(TEMPX)[0])
    
    X, Y = np.meshgrid(X1,Y1)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE,frameon='False')
    
    if arrows==True:
        #Define where to draw arrows
        SCALE = arr_scale
        WIDTH = arr_w
        ax.quiver(X, Y, (TEMPX), (TEMPY),
                  scale=SCALE,  width=WIDTH,
                  pivot='mid', color='w', alpha=0.7)
    
    EXTENT = np.min(X1), np.max(X1), np.min(Y1), np.max(Y1)
    rgba_b_reshaped = np.copy(rgba_b.reshape(np.shape(MX_NORM)[0],
                                             np.shape(MX_NORM)[1],
                                             4))
    if intensity_norm == True:
        int_norm = 255. / np.max(rgba_b_reshaped[:,:,:3])
        print('intensity_norm = ', round(int_norm, 2))
        rgba_b_reshaped[:,:,:3] = (np.copy(rgba_b_reshaped)[:,:,:3] *
                                   int_norm).astype(np.uint8)
        print(rgba_b_reshaped[0,0,:])

    ax.imshow(rgba_b_reshaped, interpolation='none',
               origin='lower', extent=EXTENT, vmin = vmin, vmax = vmax)

    if scalebar == True:
        scalebar = add_scalebar(b_inplane, pix = PIX,
                                pix_unit = pix_unit,
                                coord_system = ax.transData,
                                scalebar_fraction_of_image = 0.3,
                                loc = 'lower right', label_top = True,
                                sep = 5, frameon = False,
                                ftsize = 2*FTSIZE, color = 'white')
        ax.add_artist(scalebar)

    plt.xlabel('X [nm]', fontsize=0.85*FTSIZE)
    plt.ylabel('Y [nm]', fontsize=0.85*FTSIZE)
    plt.axis('equal')
    # if you want a title
    #plt.title(TITLE_NAME, fontsize=FTSIZE)
    if annote!='':
        PROPS = dict(boxstyle='round', facecolor='k', alpha=0.5)
        plt.text(annote_loc[0]*np.shape(MX)[1],annote_loc[1]*np.shape(MX)[0],
                 str(annote),
                 color='w',
                 fontsize=FTSIZE, bbox=PROPS)
    if im_only==True:
        ax.set_axis_off()
    plt.tight_layout()
    if sv_file_name!='':
        plt.savefig(sv_file_name+
                    '.'+str(sv_fmt),pad_inches=0,
                    dpi=300, transparent=True)
    plt.show()
    plt.clf()

def remove_outliers_dpc(image, threshold=3):
    # Compute the mean and standard deviation of the image
    image_median = np.median(image)
    image_std = np.std(image)

    # Define the lower and upper thresholds for outlier removal
    lower_threshold = image_median - threshold * image_std
    upper_threshold = image_median + threshold * image_std

    # Apply outlier removal by setting outliers to the mean value
    image_cleaned = np.copy(image)
    image_cleaned[(image < lower_threshold) | (image > upper_threshold)] = image_median

    return(image_cleaned)

def rotate_3d_vec2(mag,angx,angy,angz,pix=1,PAD=True,spins_only=False,
                   TEST=False):
    """
    This function rotates a 3D vector array with shape (3,NY_PIX,NX_PIX) in
    3 directions, with the rotation angles defined within the arrays angx,
    angy and angz in degrees, which should all be the same length. The returned
    array is not normalized.
    """
    from tqdm import tqdm
    if PAD==True:
        max_ang = np.max(np.abs(np.array([angx, angy])))/180.*np.pi
        # Determine the number of pixels to pad the array in order to avoid edges
        # within the tilted array
        LENX_ORIG = np.shape(mag)[3]
        LENY_ORIG = np.shape(mag)[2]
        X_PAD = int((LENX_ORIG/np.cos(max_ang)-LENX_ORIG+1))
        M = np.pad(np.copy(mag),pad_width=((0,0),(0,0),(X_PAD//2,X_PAD//2),
                                           (X_PAD//2,X_PAD//2)),
                   mode='edge')
    else:
        M = np.copy(mag)
    PIX=pix
    N_ANG = len(angx)
    #print(N_ANG)
    ANG = zip(angx,angy,angz)
    N_PIX_ORIG_Y = np.shape(M)[2] #pixels
    N_PIX_Y = np.shape(M)[2] #pixels
    N_PIX_ORIG_X = np.shape(M)[3] #pixels
    N_PIX_X = np.shape(M)[3] #pixels
    X_ORIG = np.linspace(0,PIX*N_PIX_X,N_PIX_ORIG_X)
    X = np.linspace(0,PIX*N_PIX_X,N_PIX_X)
    Y_ORIG = np.linspace(0,PIX*N_PIX_Y,N_PIX_ORIG_Y)
    Y = np.linspace(0,PIX*N_PIX_Y,N_PIX_Y)
    # Assuming X_ORIG, Y_ORIG, and M_TEMP are already defined
    X_mesh, Y_mesh = np.meshgrid(X, Y)  # Create a 2D grid of coordinates
    
    # Flatten the meshgrid for interpolation query
    query_points = np.vstack([Y_mesh.flatten(), X_mesh.flatten()]).T  # Combine X and Y as 2D coordinates
    BOX = slice(0,N_PIX_ORIG_Y),slice(0,N_PIX_ORIG_X)
    M_RETURN = []
    for n, ANGLE in enumerate(tqdm(ANG)):
        if TEST==True:
            print(n,ANGLE)
        M_TEMP = np.copy(M)
        M_ROT0 = np.empty((np.shape(M)),dtype=np.float64)

        #vector rotation
        for j,ANG_IT in enumerate(ANGLE):
            if TEST==True:
                print(j,ANG_IT)
            ANG_IT_rad = ANG_IT/180.*np.pi
            if ANG_IT!=0:
                for i,_ in enumerate(M[0]):
                    print(np.shape(M_TEMP), np.shape(np.copy(M_TEMP)[0][i][BOX]))
                    FX=sp.interpolate.RegularGridInterpolator((X_ORIG,Y_ORIG),
                                               np.copy(M_TEMP)[0][i][BOX])
                    FY=sp.interpolate.RegularGridInterpolator((X_ORIG,Y_ORIG),
                                               np.copy(M_TEMP)[1][i][BOX])
                    FZ=sp.interpolate.RegularGridInterpolator((X_ORIG,Y_ORIG),
                                               np.copy(M_TEMP)[2][i][BOX])
                    MX=FX(query_points).reshape(np.shape(np.copy(M_TEMP)[0][i][BOX]))
                    MY=FY(query_points).reshape(np.shape(np.copy(M_TEMP)[0][i][BOX]))
                    MZ=FZ(query_points).reshape(np.shape(np.copy(M_TEMP)[0][i][BOX]))

                    MX_NORM=(np.copy(MX))#/np.max(np.abs([MX,MY,MZ])))
                    MY_NORM=(np.copy(MY))#/np.max(np.abs([MX,MY,MZ])))
                    MZ_NORM=(np.copy(MZ))#/np.max(np.abs([MX,MY,MZ])))
                    print(np.shape(MX_NORM))
                    if j==0:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.x_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    elif j==1:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.y_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    elif j==2:
                        M_ROT0[0,i],M_ROT0[1,i],M_ROT0[2,i]=vt.z_rot_2D(MX_NORM,MY_NORM,
                                                                        MZ_NORM,
                                                                        theta=ANG_IT_rad)
                    else:
                        print("Improper axis rotation")

                for l,M_ROT in enumerate(M_ROT0):
                    M_TEMP[l] = np.copy(M_ROT)
        #image rotation
        ##for N_m,AX in enumerate(M_TEMP):
        print("ANGLE=",ANGLE)
        if spins_only==True:
            TEMP = np.copy(M_TEMP)
        else:
            TEMP = nd.rotate(nd.rotate(nd.rotate(np.copy(M_TEMP),
                                                 axes=(1,3),
                                                 angle=ANGLE[1]),
                                       axes=(1,2),angle=ANGLE[0]),
                             axes=(2,3),angle=ANGLE[2])
        if TEST==True:
            print(np.shape(TEMP))
        print(np.unique(TEMP),"0")
        print(np.unique(TEMP),"1")
        if TEST==True:
            print("np.shape(TEMP)=",np.shape(TEMP))
        LENY = np.shape(TEMP)[2]
        LENX = np.shape(TEMP)[3]
        TEMP_MIDY = LENY//2
        TEMP_MIDX = LENX//2
        MIDY = np.shape(M)[2]//2
        MIDX = np.shape(M)[3]//2
        if LENY >= np.shape(M)[2]:
            LENY = np.shape(M)[2]
        if LENX >= np.shape(M)[3]:
            LENX = np.shape(M)[3]
        if LENY >= LENY_ORIG:
            LENY = LENY_ORIG
        if LENX >= LENX_ORIG:
            LENX = LENX_ORIG
        if TEST==True:
            print("MIDY, MIDX, LENY, LENX", MIDY, MIDX, LENY, LENX)
        EXTY = LENY//2
        EXTX = LENX//2
        BOX_F = slice(MIDY-EXTY,MIDY+EXTY),slice(MIDX-EXTX,MIDX+EXTX)
        BOX_TEMP = slice(TEMP_MIDY-EXTY,
                         TEMP_MIDY+EXTY),slice(TEMP_MIDX-EXTX,
                                               TEMP_MIDX+EXTX)
        M_RETURN.append(np.copy(TEMP)[:,:,BOX_TEMP[0],BOX_TEMP[1]])
        ##M_ROTF = np.copy(M_ROTF)/np.max(np.abs(M_ROTF))
        ##print(np.unique(M_ROTF),"2")
        ##M_RETURN.append(M_ROTF)
    return M_RETURN
