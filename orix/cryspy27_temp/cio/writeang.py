# -*- coding: utf-8 -*-
# Created on Fri Feb 17 17:41:38 2017 by paytonej

def writeang(ebsd_object, filename):
    
    import cryspy.util as util
    import cryspy.rot as rot
    import io
    import numpy as np

    pb = util.progbar(finalcount=ebsd_object.nr, message='Writing to ANG')
    with io.open(filename, 'w', newline='\r\n') as fo:
        
        # FIXME: We need to fill in the appropriate stuff for the header from what
        # is stored internally in the ebsd_data object. Until then, this works for
        # data taken from *.ang files.           
        for item in ebsd_object._anghdr:
            fo.write(u'# {0:s}\n'.format(item))
        
        euler = rot.bunge.from_quat(
                                 ebsd_object.orientations.rotations).to_array()

        xy = np.vstack([ebsd_object.x, ebsd_object.y]).T
        
        # TODO: Check if we are saving image quality and ci in this way from a
        # ctf import. If we are not, what should we write here?
        iqciphase = np.vstack([ebsd_object.iq, ebsd_object.ci, 
                               ebsd_object.phaseid-1]).T
        
        num_other_cols = ebsd_object._oimversion_numangcols - 8
        
        # FIXME: What if there is no ebsd_data._other because it didn't come from a previous TSL dataset?
        oth = np.reshape(np.asanyarray(ebsd_object._other), 
                         [len(ebsd_object._other) / num_other_cols, 
                          num_other_cols])
        
        out_data = np.hstack([euler, xy, iqciphase, oth])
        
        if out_data.shape[1] == 14: # see loadosc
            fmt_str = u'{0: >9.5f} {1: >9.5f} {2: >9.5f} {3: >12.5f} '+ \
                      u'{4: >12.5f} {5: >6.1f} {6: >6.3f} {7: >2d} ' + \
                      u'{8: >6d} {9: >6.3f} {10: >9.6f} {11: >9.6f} ' + \
                      u'{12: >9.6f} {13: >9.6f} \n'  
        elif out_data.shape[1] == 10: # see loadosc
            fmt_str = u'{0: >9.5f} {1: >9.5f} {2: >9.5f} {3: >12.5f}' + \
                      u'{4: >12.5f} {5: >6.1f} {6: >6.3f} {7: >2d} ' + \
                      u'{8: >6d} {9: >6.3f} \n'
        elif out_data.shape[1] == 9: # Taken from my old writeang.m 2010-03-06 EJP
            fmt_str = u'{0: >9.3f} {1: >9.3f} {2: >9.3f} {3: >12.3f}' + \
                      u'{4: >12.3f} {5: >6.1f} {6: >6.3f} {7: >2d} ' + \
                      u'{8: >6d} \n'        
        
        for i in np.arange(0, ebsd_object.nr):
       
            p = out_data[i, :]          
            fo.write(fmt_str.format(p[0], p[1], p[2], p[3], p[4], p[5], \
                                    p[6], p[7].astype(np.int), \
                                    p[8].astype(np.int), p[9], p[10], \
                                    p[11], p[12], p[13]))
            pb.update(i)
    
    pb.update(-1)