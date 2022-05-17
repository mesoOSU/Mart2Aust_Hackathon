# -*- coding: utf-8 -*-


def loadebsd(filepath):

    '''
    Load EBSD data based on filename extension.
    '''
    test = filepath.split('.')[-1]
    if test == 'ctf':
        from cryspy.io import loadctf
        ovdat = loadctf(filepath)
    elif test == 'ang':
        from cryspy.io import loadang
        ovdat = loadang(filepath)
    elif test == 'osc':
        from cryspy.io import loadosc
        ovdat = loadosc(filepath, create_ebsd_object = True, \
                        ang_output = False, ang_output_filename = None)
    else:
        'EBSD data can be read in ctf, ang, and osc formats.'
    
    return ovdat
        
    