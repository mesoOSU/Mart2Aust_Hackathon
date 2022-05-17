# -*- coding: utf-8 -*-

def loadctf(filepath):
    
    import numpy as np
    import cryspy.ebsd as ebsd
    import cryspy.rot as rot
    import cryspy.util as util
    import cryspy.xtal as xtal

    # open the file and separate out the header from the data
    hdr=[]; dat=[]; hdr_flag = True
    with open(filepath,'r') as f:
        for line in f:
            ll = line.strip()
            if hdr_flag:
                hdr.append(ll)
            if not hdr_flag:
                dat.append(line)
            if ''.join(ll[0:9].lower().split())=='phasexy': # assume users will
                                                            # never call their
                                                            # data 'phasexy'
                hdr_flag = False
    
    # Find the locations of information to extract from header
    phaseloc=[];
    for i,str in enumerate(hdr):
        if (str.lower().find('phases\t') != -1):
            phaseloc.append(i)
    phaseloc = phaseloc[-1] # assume the last usage is the correct one, in case
                            # the user named their project something including
                            #'phases\t'
    
    # Parse phase data
    phaseidnums=[]; materialname=[]; symmetry=[];#formula=[];info=[];
    latticeconstants=[]; #numberfamilies=[];famloc=[];
    for i in range(phaseloc+1, np.shape(hdr)[0]-1):
        
        strdat = hdr[i].split()
    
        # get the lattice constants
        latticeconstants.append((strdat[0].split(';')+strdat[1].split(';')))
    
        # Get the phase number
        phaseidnums.append(i-phaseloc)
        
        # Assume that the material name always follows
        materialname.append(strdat[2])
        
        # Symmetry
        symmetry.append( 
                xtal.point_group_number_from_space_group_number(\
                                                       int(strdat[4].strip())))
    
    
    pb = util.progbar(finalcount=np.size(dat), message='LOADING EBSD DATA')
    
    # Parse the data into lists
    shdat = np.shape(dat)
    eul1=np.zeros(shdat)
    eul2=np.zeros(shdat)
    eul3=np.zeros(shdat)
    x=np.zeros(shdat)
    y=np.zeros(shdat)
    bc=np.zeros(shdat)
    bs=np.zeros(shdat)
    nbands=np.zeros(shdat)
    error=np.zeros(shdat)
    mad=np.zeros(shdat)
    phase=np.zeros(shdat)
    
    ndex = 0
    for line in dat:
        
        s= map(float, line.split())
        
        phase[ndex]  = s[ 0]
        x[ndex]      = s[ 1]
        y[ndex]      = s[ 2]
        nbands[ndex] = s[ 3]
        error[ndex] = s[ 4]
        eul1[ndex]   = s[ 5]
        eul2[ndex]   = s[ 6]
        eul3[ndex]   = s[ 7]
        mad[ndex]    = s[ 8]
        bc[ndex]     = s[ 9]
        bs[ndex]     = s[10]
        pb.update(ndex)     
        ndex += 1
        
    # Cast data lists into arrays
    phase=np.array(phase)+1 # because 0 printed for the phase when there is
                            # only one phase. but what about when there are
                            # more than one? EJP! Needs check.
    
    s = np.zeros(np.shape(phase))
    index = 0
    for item in phaseidnums:
        loc = phase == item
        s[loc] = symmetry[index]
        index += 1 
    #return s, eul1, eul2, eul3
    # Start working with what we've interpreted now
    o = xtal.orientation(quaternions = rot.quat.from_bunge(
                                                     rot.bunge(eul1,eul2,eul3)),
                         pointgroupnumbers = np.atleast_1d(s.astype(np.int)))
    pb.update(-1)
    
    ebsd_data = ebsd(orientations=o, x=x, y=y, phaseid=phase)
    ebsd_data.calc_scanstepsize()
    ebsd_data.prepare_for_plotting()

    # Add other fields. Note that some fields are CTF-specific.
    ebsd_data.bc = bc
    ebsd_data.bs = bs
    ebsd_data.mad = mad
    ebsd_data.error = error
    ebsd_data.nbands = nbands
    ebsd_data._original_header = hdr

    return ebsd_data