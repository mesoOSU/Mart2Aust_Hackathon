# -*- coding: utf-8 -*-

def loadang(filepath, alldata=False):
    
    # TODO: Add an interpreter of what version based on number of columns etc
    
    import string
    import numpy as np
    from cryspy.ebsd import ebsd
    from cryspy.rot import bunge, quat
    from cryspy.xtal import orientation, interpret_point_group_name
    from cryspy.util import progbar
    
    # open the file and separate out the header from the data
    hdr=[];dat=[];
    with open(filepath,'r') as f:
        for line in f:
            if line[0]=='#':
                hdr.append(line.strip()[1:].strip()) # remove #'s and whitespace
            else:
                dat.append(line.strip())
        
    
    # Find the locations of information to extract from header
    phaseloc=[];
    for i,istr in enumerate(hdr):
        if (string.lower(istr).find('phase') != -1):
            phaseloc.append(i)
        
        # TODO: THE FOLLOWING ARE NOT YET NEEDED IN THE CRYSPY CODE, BUT
        #       ARE NEEDED FOR WRITING ANG FILES
        
        #nom='TEM_PIXperUM'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    tem_pix_per_um=float(str[(loc+len(nom)):].strip())
    
        #nom='x-star'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    x_star=float(str[(loc+len(nom)):].strip())
    
        #nom='y-star'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    y_star=float(str[(loc+len(nom)):].strip())
    
        #nom='z-star'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    z_star=float(str[(loc+len(nom)):].strip())
    
        #nom='workingdistance'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    workingdistance=float(str[(loc+len(nom)):].strip())
    
        #nom='grid:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    grid=str[(loc+len(nom)):].strip()
            
        #nom='xstep:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    xstp=float(str[(loc+len(nom)):].strip())
    
        #nom='ystep:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    ystp=float(str[(loc+len(nom)):].strip())
    
        #nom='ncols_odd:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    ncols_odd=float(str[(loc+len(nom)):].strip())
    
        #nom='ncols_even:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    ncols_even=float(str[(loc+len(nom)):].strip())
    
        #nom='nrows:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    ncols_even=float(str[(loc+len(nom)):].strip())
    
        #nom='operator:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    operator=str[(loc+len(nom)):].strip()
    
        #nom='sampleid:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    sampleid=str[(loc+len(nom)):].strip()
    
        #nom='scanid:'
        #loc=lower(str).find(nom)
        #if (loc != -1):
        #    scanid=str[(loc+len(nom)):].strip()
    
    
    # Parse phase data
    phaseidnums=[];materialname=[];formula=[];info=[];symmetry=[];latticeconstants=[];
    numberfamilies=[];famloc=[];
    for i in phaseloc:
        # Get the phase number
        nom='phase'
        str=hdr[i];loc=string.lower(str).find(nom)
        phaseidnums.append(float(str[loc+len(nom):].strip()))
        
        # Assume that the material name always follows
        nom='materialname'
        str=hdr[i+1];loc=string.lower(str).find(nom)
        materialname.append(str[loc+len(nom):].strip())
    
        # Assume that the formula is always next
        nom='formula'
        str=hdr[i+2];loc=string.lower(str).find(nom)
        formula.append(str[loc+len(nom):].strip())
        
        # Assume that the info is always after the formula...
        nom='info'
        str=hdr[i+3];loc=string.lower(str).find(nom)
        info.append(str[loc+len(nom):].strip())
    
        # ... and then the symmetry
        nom='symmetry'
        str=hdr[i+4];loc=string.lower(str).find(nom)
        symmetry.append(str[loc+len(nom):].strip())
        
        # ... and then the lattice constants
        nom='latticeconstants'
        str=hdr[i+5];loc=string.lower(str).find(nom)
        latticeconstants.append(str[loc+len(nom):].strip())
        
         # ... and then the number of families
        nom='numberfamilies'
        str=hdr[i+6];loc=string.lower(str).find(nom)
        numberfamilies.append(float(str[loc+len(nom):].strip()))
        
        # store the line location of the beginning of the families
        famloc.append(i+7)
    
    # We should obtain the grid, the x step, the y step, the numbers of 
    # columns, and the number of rows from the x and y data themselves, 
    # because WE CAN... just in case the header was copied and pasted 
    # from a different file. Then compare to what the header says and 
    # print something if there is a discrepancy
    
    pb = progbar(finalcount=np.size(dat), message='LOADING EBSD DATA')
    # Parse the data into lists

    eul1=np.zeros(np.shape(dat))
    eul2=np.zeros(np.shape(dat))
    eul3=np.zeros(np.shape(dat))
    x=np.zeros(np.shape(dat))
    y=np.zeros(np.shape(dat))
    iq=np.zeros(np.shape(dat))
    ci=np.zeros(np.shape(dat))
    phase=np.zeros(np.shape(dat))
    other = [] # XXX: other = np.array([]) # see comments below

    if alldata==True:
        
        ndex = 0
        for line in dat:
            
            s= map(float, line.split())
            
            eul1[ndex] = s[0]
            eul2[ndex] = s[1]
            eul3[ndex] = s[2]
            x[ndex] = s[3]
            y[ndex] = s[4]
            iq[ndex] = s[5]
            ci[ndex] = s[6]
            phase[ndex] = s[7]
            pb.update(ndex)     
            ndex += 1   
            
            # Throw the other data into an "other" group of flexible size
            for i in range(8, len(s)):
                #other = np.hstack([other, np.array([np.float(s[i])])]).T # FIXME: This adds a LOT of time!!! There must be a faster way to do this.
                other.append(s[i])

    else:

        ndex = 0
        for line in dat:
            
            s= map(float, line.split())
            
            eul1[ndex] = s[0]
            eul2[ndex] = s[1]
            eul3[ndex] = s[2]
            x[ndex] = s[3]
            y[ndex] = s[4]
            iq[ndex] = s[5]
            ci[ndex] = s[6]
            phase[ndex] = s[7]
            pb.update(ndex)     
            ndex += 1   

    numangcols = len(s)
        
    # Cast data lists into arrays
    phase=np.array(phase)+1 # because tsl prints 0 for the phase when there is
                            # only one phase. But what about when there are 
                            # more than one? 
    
    s = np.zeros(np.shape(phase))
    index = 0
    for item in phaseidnums:
        loc = phase == item
        s[loc] = interpret_point_group_name(symmetry[index], 'tsl')
        index += 1 
    # Start working with what we've interpreted now
    o = orientation(quaternions = quat.from_bunge(bunge(eul1,eul2,eul3)),\
                    pointgroupnumbers = np.squeeze(s.astype(np.int)))
    pb.update(-1)
    
    ebsd_data = ebsd(orientations=o, x=x, y=y, phaseid=phase)
    ebsd_data.calc_scanstepsize()
    ebsd_data.prepare_for_plotting()
    
    # Add other fields. Note that some fields are TSL-specific, and that two of
    # the fields (fit and ...) have not been added yet.
    ebsd_data.iq = iq
    ebsd_data.ci = ci
    
    ebsd_data._anghdr = u''.join(['# {0}\n'.format(line) for line in (hdr)])
    ebsd_data._originalheaderdata = hdr
    ebsd_data._other = other
    ebsd_data._oimversion_numangcols = numangcols
    
    return ebsd_data
