# -*- coding: utf-8 -*-
"""
ovlib.vis: Visualization
===============================================================================
"""

# TODO: hide all these imports in the classes and methods
from cryspy.xtal import lattvec, miller, ipfgrid, fundzonePF, symm,\
                       unitcell, interpret_point_group_name
from cryspy.util import stereotrans, revstereotrans, revstereotransstandard, \
                       vecarraynorm, vecarraydot, vecarraycross, radians,\
                       rationalize, eatrans, reveatransstandard, reveatrans, \
                       polar, cart, vecarrayconvert, tic, toc
from cryspy.ebsd import triverts
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as mplcollections
import matplotlib.delaunay as mpltriang
#import visvis
import inspect

class stereoproj(object):
    '''
    stereoproj()
    
    A stereographic projection represents the planes and the directions of a crystal.
    A pole figure is a stereographic projection of the planes in the figure.
    '''
    
    def __init__(self, figure=None, subplot=None, 
                 center=[0,0,1], south=[0,1,0], east=[1,0,0], **kwargs):
        
        if not 'figsize' in kwargs:
            kwargs['figsize'] = (8, 8)
        
        # Open the figure
        if figure:
            self._figure = figure
        else:
            self._figure = plt.figure(**kwargs)
            
        # Create first axis instance
        if subplot:
            self._ax = self._figure.add_subplot(subplot)
            self._subplot = subplot
        else:
            self._ax = self._figure.add_subplot(111)
            self._subplot = 111
        
        # Create the background
        plt.axes(self._ax).set_xbound(-1.05,1.05)
        plt.axes(self._ax).set_ybound(-1.05,1.05)
        circ=plt.Circle((0,0),1,facecolor='w', 
                        edgecolor='k', linewidth=1, alpha=1, clip_on=False)
        self._ax.add_patch(circ)

        plt.axes(self._ax).set_aspect('equal')
        plt.axes(self._ax).axison = False
    
        south = vecarrayconvert(south)
        east = vecarrayconvert(east)
        center = vecarrayconvert(center)
                
        if  abs(vecarraydot(south,east))<np.sqrt(np.spacing(1)) and \
            abs(vecarraydot(east,center))<np.sqrt(np.spacing(1)) and \
            abs(vecarraydot(center,south))<np.sqrt(np.spacing(1)):
                      
            self._center=center
            self._south=south
            self._east=east
        else:
            print('The pole figure center, south, and east directions must be \
                orthonormal. \
                Proceeding with a standard projection')
            self._center=[0,0,1]
            self._south=[0,1,0]
            self._east=[1,0,0]

        if  vecarraydot(vecarraycross(south,east),center)<np.sqrt(np.spacing(1)):
                      
            self._center=center
            self._south=south
            self._east=east
        else:
            print('The pole figure center, south, and east directions must \
                form a right-handed system. \
                Proceeding with a standard projection')
            self._center=[0,0,1]
            self._south=[0,1,0]
            self._east=[1,0,0]

#-------------------------------------------------------------------------------

    # Method for adding directions to the pole figure
    def add_lattvec(self,v,uc,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=None,
                    upperlinestyle='none',
                    uppercolor='r',
                    uppermarker='o',
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='r',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=None,
                    lowerlinestyle='none',
                    lowercolor='none',
                    lowermarker='o',
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='r',
                    lowermarkerfacecolor='r',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                     
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    **kwargs):
        
        # assign relevant defaults from mpl.rcParams
        if upperlinewidth is None : 
            upperlinewidth=mpl.rcParams['lines.linewidth']
        if upperlinestyle is None : 
            upperlinestyle=mpl.rcParams['lines.linestyle']
        if uppermarker is None : 
            uppermarker=mpl.rcParams['lines.marker']
        if uppercolor is None : 
            uppercolor=mpl.rcParams['lines.color']
        if uppermarkersize is None : 
            uppermarkersize=mpl.rcParams['lines.markersize']
        if upperdash_capstyle is None : 
            upperdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if upperdash_joinstyle is None : 
            upperdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if uppersolid_capstyle is None : 
            uppersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if uppersolid_joinstyle is None : 
            uppersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if lowerlinewidth is None : 
            lowerlinewidth=mpl.rcParams['lines.linewidth']
        if lowerlinestyle is None : 
            lowerlinestyle=mpl.rcParams['lines.linestyle']
        if lowermarker is None : 
            lowermarker=mpl.rcParams['lines.marker']
        if lowercolor is None : 
            lowercolor=mpl.rcParams['lines.color']
        if lowermarkersize is None : 
            lowermarkersize=mpl.rcParams['lines.markersize']
        if lowerdash_capstyle is None : 
            lowerdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if lowerdash_joinstyle is None : 
            lowerdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if lowersolid_capstyle is None : 
            lowersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if lowersolid_joinstyle is None : 
            lowersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']
        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']

        if isinstance(v,lattvec):
            
            plt.subplot(self._subplot)
            
            x,y,hem = stereotrans(v.to_cartesian(uc))
            
            plt.plot(x[hem=='N'],y[hem=='N'],
                     linewidth=upperlinewidth,
                     linestyle=upperlinestyle,
                     color=uppercolor,
                     marker=uppermarker,
                     markersize=uppermarkersize,
                     markeredgewidth=uppermarkeredgewidth,
                     markeredgecolor=uppermarkeredgecolor,
                     markerfacecolor=uppermarkerfacecolor,
                     markerfacecoloralt=uppermarkerfacecoloralt,
                     fillstyle=upperfillstyle,
                     dash_capstyle=upperdash_capstyle,
                     solid_capstyle=uppersolid_capstyle,
                     dash_joinstyle=upperdash_joinstyle,
                     solid_joinstyle=uppersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)
                     
            plt.plot(x[hem=='S'],y[hem=='S'],                     
                     linewidth=lowerlinewidth,
                     linestyle=lowerlinestyle,
                     color=lowercolor,
                     marker=lowermarker,
                     markersize=lowermarkersize,
                     markeredgewidth=lowermarkeredgewidth,
                     markeredgecolor=lowermarkeredgecolor,
                     markerfacecolor=lowermarkerfacecolor,
                     markerfacecoloralt=lowermarkerfacecoloralt,
                     fillstyle=lowerfillstyle,
                     dash_capstyle=lowerdash_capstyle,
                     solid_capstyle=lowersolid_capstyle,
                     dash_joinstyle=lowerdash_joinstyle,
                     solid_joinstyle=lowersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)

            plt.axes(self._ax).set_xbound(-1.05,1.05)
            plt.axes(self._ax).set_ybound(-1.05,1.05)

#----------------------------------------------------------------------------

    # Method for adding planes to the pole figure
    def add_miller(self,v,uc,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=None,
                    upperlinestyle='none',
                    uppercolor='b',
                    uppermarker='s',
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='b',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=None,
                    lowerlinestyle='none',
                    lowercolor='none',
                    lowermarker='s',
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='b',
                    lowermarkerfacecolor='b',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                    
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    **kwargs):

        plt.subplot(self._subplot)
        
        # assign relevant defaults from mpl.rcParams
        if upperlinewidth is None : 
            upperlinewidth=mpl.rcParams['lines.linewidth']
        if upperlinestyle is None : 
            upperlinestyle=mpl.rcParams['lines.linestyle']
        if uppermarker is None : 
            uppermarker=mpl.rcParams['lines.marker']
        if uppercolor is None : 
            uppercolor=mpl.rcParams['lines.color']
        if uppermarkersize is None : 
            uppermarkersize=mpl.rcParams['lines.markersize']
        if upperdash_capstyle is None : 
            upperdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if upperdash_joinstyle is None : 
            upperdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if uppersolid_capstyle is None : 
            uppersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if uppersolid_joinstyle is None : 
            uppersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if lowerlinewidth is None : 
            lowerlinewidth=mpl.rcParams['lines.linewidth']
        if lowerlinestyle is None : 
            lowerlinestyle=mpl.rcParams['lines.linestyle']
        if lowermarker is None : 
            lowermarker=mpl.rcParams['lines.marker']
        if lowercolor is None : 
            lowercolor=mpl.rcParams['lines.color']
        if lowermarkersize is None : 
            lowermarkersize=mpl.rcParams['lines.markersize']
        if lowerdash_capstyle is None : 
            lowerdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if lowerdash_joinstyle is None : 
            lowerdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if lowersolid_capstyle is None : 
            lowersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if lowersolid_joinstyle is None : 
            lowersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']

        if isinstance(v,miller):
            
            x,y,hem = stereotrans(v.to_cartesian(uc), center=self._center,
                              east=self._east, south=self._south)
            
            plt.plot(x[hem=='N'],y[hem=='N'],
                     linewidth=upperlinewidth,
                     linestyle=upperlinestyle,
                     color=uppercolor,
                     marker=uppermarker,
                     markersize=uppermarkersize,
                     markeredgewidth=uppermarkeredgewidth,
                     markeredgecolor=uppermarkeredgecolor,
                     markerfacecolor=uppermarkerfacecolor,
                     markerfacecoloralt=uppermarkerfacecoloralt,
                     fillstyle=upperfillstyle,
                     dash_capstyle=upperdash_capstyle,
                     solid_capstyle=uppersolid_capstyle,
                     dash_joinstyle=upperdash_joinstyle,
                     solid_joinstyle=uppersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)
                     
            plt.plot(x[hem=='S'],y[hem=='S'],                     
                     linewidth=lowerlinewidth,
                     linestyle=lowerlinestyle,
                     color=lowercolor,
                     marker=lowermarker,
                     markersize=lowermarkersize,
                     markeredgewidth=lowermarkeredgewidth,
                     markeredgecolor=lowermarkeredgecolor,
                     markerfacecolor=lowermarkerfacecolor,
                     markerfacecoloralt=lowermarkerfacecoloralt,
                     fillstyle=lowerfillstyle,
                     dash_capstyle=lowerdash_capstyle,
                     solid_capstyle=lowersolid_capstyle,
                     dash_joinstyle=lowerdash_joinstyle,
                     solid_joinstyle=lowersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)

        else:
            print 'add_miller method for pole figures only accepts miller ' + \
                  'class input'

#----------------------------------------------------------------------------

    def add_trace(self,v,uc,
                    resolution=25,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=1,
                    upperlinestyle='-',
                    uppercolor='b',
                    uppermarker=None,
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='none',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=1,
                    lowerlinestyle='--',
                    lowercolor='b',
                    lowermarker=None,
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='none',
                    lowermarkerfacecolor='none',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                    
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    zorder=2,
                    **kwargs):

        plt.subplot(self._subplot)
               
        # assign relevant defaults from mpl.rcParams
        if upperlinewidth is None : 
            upperlinewidth=mpl.rcParams['lines.linewidth']
        if upperlinestyle is None : 
            upperlinestyle=mpl.rcParams['lines.linestyle']
        if uppermarker is None : 
            uppermarker=mpl.rcParams['lines.marker']
        if uppercolor is None : 
            uppercolor=mpl.rcParams['lines.color']
        if uppermarkersize is None : 
            uppermarkersize=mpl.rcParams['lines.markersize']
        if upperdash_capstyle is None : 
            upperdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if upperdash_joinstyle is None : 
            upperdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if uppersolid_capstyle is None : 
            uppersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if uppersolid_joinstyle is None : 
            uppersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if lowerlinewidth is None : 
            lowerlinewidth=mpl.rcParams['lines.linewidth']
        if lowerlinestyle is None : 
            lowerlinestyle=mpl.rcParams['lines.linestyle']
        if lowermarker is None : 
            lowermarker=mpl.rcParams['lines.marker']
        if lowercolor is None : 
            lowercolor=mpl.rcParams['lines.color']
        if lowermarkersize is None : 
            lowermarkersize=mpl.rcParams['lines.markersize']
        if lowerdash_capstyle is None : 
            lowerdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if lowerdash_joinstyle is None : 
            lowerdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if lowersolid_capstyle is None : 
            lowersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if lowersolid_joinstyle is None : 
            lowersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']
        
        if isinstance(v,miller):
            
            abc = vecarrayconvert(self._center)
            nrmabc = 1.0/vecarraynorm(abc)    
            
            uvw = vecarrayconvert(self._east)
            nrmuvw = 1.0/vecarraynorm(uvw)
            
            fed = vecarrayconvert(self._south) # 'def' is reserved in python
            nrmdef = 1.0/vecarraynorm(fed) 
            
            xyz = v.to_cartesian(uc)
            nrmxyz = 1.0/vecarraynorm(xyz)
            
            xx = xyz[0]*nrmxyz
            yy = xyz[1]*nrmxyz
            zz = xyz[2]*nrmxyz
        
            n = np.shape(xx)     
            
            aa = np.tile(abc[0]*nrmabc,n)
            bb = np.tile(abc[1]*nrmabc,n)
            cc = np.tile(abc[2]*nrmabc,n)
            dd = np.tile(fed[0]*nrmdef,n)
            ee = np.tile(fed[1]*nrmdef,n)
            ff = np.tile(fed[2]*nrmdef,n)
            uu = np.tile(uvw[0]*nrmuvw,n)
            vv = np.tile(uvw[1]*nrmuvw,n)
            ww = np.tile(uvw[2]*nrmuvw,n)
        
            cosdl = vecarraydot([xx,yy,zz],[dd,ee,ff])
            cosmu = vecarraydot([xx,yy,zz],[uu,vv,ww])
            cosal = vecarraydot([xx,yy,zz],[aa,bb,cc])
            
            denom = 1.0/(1.0+abs(cosal))
            
            xproj = cosmu * denom
            yproj = cosdl * denom
        
            hemis = np.tile('N',n)    
            if np.array(n)>1:
                hemis[cosal<0.0] = 'S'
            else:
                if cosal<0.0:
                    hemis = np.tile('S',n)
            
            denom = np.sqrt(xproj*xproj+yproj*yproj)
            denom[denom==0.0]=1.0
            cosrho = xproj/denom
            sinrho = yproj/denom
            cosrho[denom==0.0]=0.0
            sinrho[denom==0.0]=0.0
            
            alpha = np.arccos(cosal)
    
    
            i=-1
            for almx in alpha:
                
                # 1/cos and tan have problems near pi/2
                if (abs(almx)-np.pi/2.0)<np.sqrt(np.spacing(1)):
                    almx=abs(almx)-np.sqrt(np.spacing(1))
                    
                
                rgc = 1.0/np.cos(almx)
                ogc = np.tan(almx)
                
                thi = np.pi/2.0+almx
                thf = 3.0*np.pi/2.0-almx
                ths = (thf-thi)/resolution
    
                th  = np.arange(thi,thf+ths/2.0,ths)
                x = rgc*np.cos(th)+ogc
                y = rgc*np.sin(th)

                i=i+1                
                cr = cosrho[i]
                sr = sinrho[i]

                if hemis[i]=='N':
                    plt.plot(x*cr-y*sr,y*cr+x*sr,
                             linewidth=upperlinewidth,
                             linestyle=upperlinestyle,
                             color=uppercolor,
                             marker=uppermarker,
                             markersize=uppermarkersize,
                             markeredgewidth=uppermarkeredgewidth,
                             markeredgecolor=uppermarkeredgecolor,
                             markerfacecolor=uppermarkerfacecolor,
                             markerfacecoloralt=uppermarkerfacecoloralt,
                             fillstyle=upperfillstyle,
                             dash_capstyle=upperdash_capstyle,
                             solid_capstyle=uppersolid_capstyle,
                             dash_joinstyle=upperdash_joinstyle,
                             solid_joinstyle=uppersolid_joinstyle,
                             pickradius=pickradius,
                             drawstyle=drawstyle,
                             markevery=markevery,
                             antialiased=antialiased,
                             zorder=zorder,
                             **kwargs)
                             
                if hemis[i]=='S':
                    plt.plot(-x*cr+y*sr,-y*cr-x*sr,
                             linewidth=lowerlinewidth,
                             linestyle=lowerlinestyle,
                             color=lowercolor,
                             marker=lowermarker,
                             markersize=lowermarkersize,
                             markeredgewidth=lowermarkeredgewidth,
                             markeredgecolor=lowermarkeredgecolor,
                             markerfacecolor=lowermarkerfacecolor,
                             markerfacecoloralt=lowermarkerfacecoloralt,
                             fillstyle=lowerfillstyle,
                             dash_capstyle=lowerdash_capstyle,
                             solid_capstyle=lowersolid_capstyle,
                             dash_joinstyle=lowerdash_joinstyle,
                             solid_joinstyle=lowersolid_joinstyle,
                             pickradius=pickradius,
                             drawstyle=drawstyle,
                             markevery=markevery,
                             antialiased=antialiased,
                             zorder=zorder,
                             **kwargs)                             

#----------------------------------------------------------------------------

    def grid_spokes(self,degreestep=90.0,mindegrees=0.0,maxdegrees=360.0,
                     resolution=25,
                     poleclip=0,
                     rotation=0.0,
                     linewidth=1,
                     linestyle='-',
                     color=[0.6,0.6,0.6],
                     marker=None,
                     markersize=None,
                     markeredgewidth='none',
                     markeredgecolor='none',
                     markerfacecolor='none',
                     markerfacecoloralt='none',
                     fillstyle='full',
                     dash_capstyle=None,
                     solid_capstyle=None,
                     dash_joinstyle=None,
                     solid_joinstyle=None,
                     pickradius=None,
                     drawstyle=None,
                     markevery=None,
                     antialiased=None,
                     zorder=1,
                     **kwargs):

        plt.subplot(self._subplot)

        # assign relevant defaults from mpl.rcParams
        if linewidth is None : 
            linewidth=mpl.rcParams['lines.linewidth']
        if linestyle is None : 
            linestyle=mpl.rcParams['lines.linestyle']
        if marker is None : 
            marker=mpl.rcParams['lines.marker']
        if color is None : 
            color=mpl.rcParams['lines.color']
        if markersize is None : 
            markersize=mpl.rcParams['lines.markersize']
        if dash_capstyle is None : 
            dash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if dash_joinstyle is None : 
            dash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if solid_capstyle is None : 
            solid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if solid_joinstyle is None : 
            solid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']

        poleclip=radians(poleclip)
        
        th = np.arange(mindegrees,maxdegrees+degreestep,degreestep)
        th = th * np.pi / 180.0
        lin = np.arange(poleclip,1.0+0.5/resolution,1.0/resolution)
        
        rho = radians(rotation)
        cosrho = np.cos(rho)
        sinrho = np.sin(rho)
        
        for alph in th:
            
            x = lin * np.cos(alph)
            y = lin * np.sin(alph)
              
            plt.plot(x*cosrho-y*sinrho,y*cosrho+x*sinrho,
                     linewidth=linewidth,
                     linestyle=linestyle,
                     color=color,
                     marker=marker,
                     markersize=markersize,
                     markeredgewidth=markeredgewidth,
                     markeredgecolor=markeredgecolor,
                     markerfacecolor=markerfacecolor,
                     markerfacecoloralt=markerfacecoloralt,
                     fillstyle=fillstyle,
                     dash_capstyle=dash_capstyle,
                     solid_capstyle=solid_capstyle,
                     dash_joinstyle=dash_joinstyle,
                     solid_joinstyle=solid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     zorder=zorder,
                     **kwargs)
            
            plt.axes(plt.gca()).set_xbound(-1.05,1.05)
            plt.axes(plt.gca()).set_ybound(-1.05,1.05)

#----------------------------------------------------------------------------

    def grid_greatcircles(self,degreestep=45.0,
                                 mindegrees=0.0,maxdegrees=180,
                                 resolution=25,
                                 linewidth=1,
                                 linestyle='-',
                                 color=[0.6,0.6,0.6],
                                 poleclip=0.0,
                                 rotation=0.0,
                                 marker=None,
                                 markersize=None,
                                 markeredgewidth='none',
                                 markeredgecolor='none',
                                 markerfacecolor='none',
                                 markerfacecoloralt='none',
                                 fillstyle='full',
                                 dash_capstyle=None,
                                 solid_capstyle=None,
                                 dash_joinstyle=None,
                                 solid_joinstyle=None,
                                 pickradius=None,
                                 drawstyle=None,
                                 markevery=None,
                                 antialiased=None,
                                 zorder=1,
                                 **kwargs):

        plt.subplot(self._subplot)
                                     
        # assign relevant defaults from mpl.rcParams
        if linewidth is None : 
            linewidth=mpl.rcParams['lines.linewidth']
        if linestyle is None : 
            linestyle=mpl.rcParams['lines.linestyle']
        if marker is None : 
            marker=mpl.rcParams['lines.marker']
        if color is None : 
            color=mpl.rcParams['lines.color']
        if markersize is None : 
            markersize=mpl.rcParams['lines.markersize']
        if dash_capstyle is None : 
            dash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if dash_joinstyle is None : 
            dash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if solid_capstyle is None : 
            solid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if solid_joinstyle is None : 
            solid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']
        
        # Create the alpha range        
        alpha = np.arange(mindegrees,maxdegrees+degreestep,degreestep)
        if any(alpha>180.0) or any(alpha<0.0):
            print 'valid range for grid_verticalgreatcircles \
                   is between 0 and 180'

        # Put data in our valid range
        alpha = alpha * np.pi/180.0            
        alpha = alpha[alpha>0.0]
        alpha = alpha[alpha<np.pi]
        alpha = alpha[alpha!=np.pi/2.0]
        poleclip = np.pi/2.0-radians(poleclip)+np.spacing(1)
        
        rho = radians(rotation)
        cosrho = np.cos(rho)
        sinrho = np.sin(rho)
        
        # clip near the poles using the geometry of 
        # the small circle and the chord formula
        rsc = 1.0/np.tan(poleclip)
        osc = 1.0/np.sin(poleclip)
        
        for almx in alpha:
            if almx<np.pi/2:
                rgc = 1.0/np.cos(almx)
                ogc = np.tan(almx)
                d2 = osc * osc + ogc * ogc
                kk  = 0.25 * np.sqrt((( rsc + rgc)**2.0 - d2) \
                        * (d2 - (rgc - rsc)**2.0 ))
                x2 = 0.5*ogc - 0.5*ogc*(rgc*rgc - rsc*rsc)/d2 - 2.0*osc*kk/d2
                y2 = 0.5*osc + 0.5*osc*(rgc*rgc - rsc*rsc)/d2 - 2.0*ogc*kk/d2
                v1 = np.array([-ogc, 1.0])
                v1 = v1 / npla.norm(v1)
                v2 = np.array([x2 - ogc, y2])
                v2 = v2 / npla.norm(v2)
                pc = np.arccos(np.dot(v1, v2))
                
                thi = np.pi/2.0+almx+pc
                thf = 3.0*np.pi/2.0-almx-pc
                ths = (thf-thi)/resolution

                th  = np.arange(thi,thf+0.5*ths,ths)
                x = rgc*np.cos(th)+ogc
                y = rgc*np.sin(th)
                
                rho=radians(25.0)
                plt.plot(x*cosrho-y*sinrho,y*cosrho+x*sinrho,
                         linewidth=linewidth,
                         linestyle=linestyle,
                         color=color,
                         marker=marker,
                         markersize=markersize,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor,
                         markerfacecolor=markerfacecolor,
                         markerfacecoloralt=markerfacecoloralt,
                         fillstyle=fillstyle,
                         dash_capstyle=dash_capstyle,
                         solid_capstyle=solid_capstyle,
                         dash_joinstyle=dash_joinstyle,
                         solid_joinstyle=solid_joinstyle,
                         pickradius=pickradius,
                         drawstyle=drawstyle,
                         markevery=markevery,
                         antialiased=antialiased,
                         zorder=zorder, #place b/t grid & background
                         **kwargs)
            
            if almx>=np.pi/2:
                rgc = 1.0/np.cos(almx)
                ogc = np.tan(almx)
                d2 = osc * osc + ogc * ogc
                kk  = 0.25 * np.sqrt((( rsc + rgc)**2.0 - d2) \
                        * (d2 - (rgc - rsc)**2.0 ))
                x2 = 0.5*ogc-0.5*ogc*(rgc*rgc - rsc*rsc)/d2 + 2.0*osc*kk/d2
                y2 = 0.5*osc+0.5*osc*(rgc*rgc - rsc*rsc)/d2 + 2.0*ogc*kk/d2
                v1=np.array([-ogc,1.0])
                v1=v1/npla.norm(v1)
                v2=np.array([x2-ogc,y2])
                v2=v2/npla.norm(v2)
                pc=np.arccos(np.dot(v1, v2))             
                
                thi=3.0*np.pi/2.0-almx+pc
                thf=np.pi/2.0+almx-pc
                ths=(thf-thi)/resolution
                
                th  = np.arange(thi,thf+0.5*ths,ths)
                x = rgc*np.cos(th)+ogc
                y = rgc*np.sin(th)
                plt.plot(x*cosrho-y*sinrho,y*cosrho+x*sinrho,
                         linewidth=linewidth,
                         linestyle=linestyle,
                         color=color,
                         marker=marker,
                         markersize=markersize,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor,
                         markerfacecolor=markerfacecolor,
                         markerfacecoloralt=markerfacecoloralt,
                         fillstyle=fillstyle,
                         dash_capstyle=dash_capstyle,
                         solid_capstyle=solid_capstyle,
                         dash_joinstyle=dash_joinstyle,
                         solid_joinstyle=solid_joinstyle,
                         pickradius=pickradius,
                         drawstyle=drawstyle,
                         markevery=markevery,
                         antialiased=antialiased,
                         zorder=zorder,
                         **kwargs)
                
            plt.axes(plt.gca()).set_xbound(-1.05,1.05)
            plt.axes(plt.gca()).set_ybound(-1.05,1.05)

#----------------------------------------------------------------------------

    def grid_smallcircles(self,degreestep=45.0,
                                 mindegrees=0.0,maxdegrees=180,
                                 resolution=25,
                                 rotation=0.0,
                                 linewidth=1,
                                 linestyle='-',
                                 color=[0.6,0.6,0.6],
                                 marker=None,
                                 markersize=None,
                                 markeredgewidth='none',
                                 markeredgecolor='none',
                                 markerfacecolor='none',
                                 markerfacecoloralt='none',
                                 fillstyle='full',
                                 dash_capstyle=None,
                                 solid_capstyle=None,
                                 dash_joinstyle=None,
                                 solid_joinstyle=None,
                                 pickradius=None,
                                 drawstyle=None,
                                 markevery=None,
                                 antialiased=None,
                                 zorder=1,
                                 **kwargs):

        plt.subplot(self._subplot)
                                     
        # assign relevant defaults from mpl.rcParams
        if linewidth is None : 
            linewidth=mpl.rcParams['lines.linewidth']
        if linestyle is None : 
            linestyle=mpl.rcParams['lines.linestyle']
        if marker is None : 
            marker=mpl.rcParams['lines.marker']
        if color is None : 
            color=mpl.rcParams['lines.color']
        if markersize is None : 
            markersize=mpl.rcParams['lines.markersize']
        if dash_capstyle is None : 
            dash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if dash_joinstyle is None : 
            dash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if solid_capstyle is None : 
            solid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if solid_joinstyle is None : 
            solid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']
        
        # Create the alpha range        
        beta = np.arange(mindegrees,maxdegrees+degreestep,degreestep)
        if any(beta>180.0) or any(beta<0.0):
            print 'valid range for grid_verticalgreatcircles \
                   is between 0 and 180'

        # Put data in our valid range
        beta = beta * np.pi/180.0            
        beta = beta[beta>0.0]
        beta = beta[beta<np.pi]
        beta = beta[beta!=np.pi/2.0] 
        
        rho = radians(rotation)
        cosrho = np.cos(rho)
        sinrho = np.sin(rho)
        
        for bemx in beta:
            if bemx<np.pi/2.0:
                rsc = 1.0/np.tan(bemx)
                org = 1.0/np.sin(bemx)
                
                thi = np.pi/2.0-bemx
                thf = np.pi/2.0+bemx
                ths = (thf-thi)/resolution
                th = np.arange(thi,thf+0.5*ths,ths)              
                
                x = rsc*np.cos(th)
                y = rsc*np.sin(th)-org
                plt.plot(x*cosrho-y*sinrho,y*cosrho+x*sinrho,
                         linewidth=linewidth,
                         linestyle=linestyle,
                         color=color,
                         marker=marker,
                         markersize=markersize,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor,
                         markerfacecolor=markerfacecolor,
                         markerfacecoloralt=markerfacecoloralt,
                         fillstyle=fillstyle,
                         dash_capstyle=dash_capstyle,
                         solid_capstyle=solid_capstyle,
                         dash_joinstyle=dash_joinstyle,
                         solid_joinstyle=solid_joinstyle,
                         pickradius=pickradius,
                         drawstyle=drawstyle,
                         markevery=markevery,
                         antialiased=antialiased,
                         zorder=zorder,# place grid b/t data and background
                         **kwargs)
            
            if bemx>=np.pi/2:
                rsc = 1.0/np.tan(bemx)
                org = 1.0/np.sin(bemx)
                thi = bemx-np.pi/2.0
                thf = 3.0*np.pi/2.0-bemx
                ths = (thf-thi)/resolution
                th = -np.arange(thi,thf+0.5*ths,ths)
                
                x = rsc*np.cos(th)
                y = -rsc*np.sin(th)+org
                plt.plot(x*cosrho-y*sinrho,y*cosrho+x*sinrho,
                         linewidth=linewidth,
                         linestyle=linestyle,
                         color=color,
                         marker=marker,
                         markersize=markersize,
                         markeredgewidth=markeredgewidth,
                         markeredgecolor=markeredgecolor,
                         markerfacecolor=markerfacecolor,
                         markerfacecoloralt=markerfacecoloralt,
                         fillstyle=fillstyle,
                         dash_capstyle=dash_capstyle,
                         solid_capstyle=solid_capstyle,
                         dash_joinstyle=dash_joinstyle,
                         solid_joinstyle=solid_joinstyle,
                         pickradius=pickradius,
                         drawstyle=drawstyle,
                         markevery=markevery,
                         antialiased=antialiased,
                         zorder=zorder,
                         **kwargs)
                
            plt.axes(plt.gca()).set_xbound(-1.05,1.05)
            plt.axes(plt.gca()).set_ybound(-1.05,1.05)

#----------------------------------------------------------------------------

    def grid_rings(self,degreestep=45.0,
                                 mindegrees=0.0,maxdegrees=90.0,
                                 resolution=25,
                                 linewidth=1,
                                 linestyle='-',
                                 color=[0.6,0.6,0.6],
                                 marker=None,
                                 markersize=None,
                                 markeredgewidth='none',
                                 markeredgecolor='none',
                                 markerfacecolor='none',
                                 markerfacecoloralt='none',
                                 fillstyle='full',
                                 dash_capstyle=None,
                                 solid_capstyle=None,
                                 dash_joinstyle=None,
                                 solid_joinstyle=None,
                                 pickradius=None,
                                 drawstyle=None,
                                 markevery=None,
                                 antialiased=None,
                                 zorder=1,
                                 **kwargs):

        plt.subplot(self._subplot)
                                     
        # assign relevant defaults from mpl.rcParams
        if linewidth is None : 
            linewidth=mpl.rcParams['lines.linewidth']
        if linestyle is None : 
            linestyle=mpl.rcParams['lines.linestyle']
        if marker is None : 
            marker=mpl.rcParams['lines.marker']
        if color is None : 
            color=mpl.rcParams['lines.color']
        if markersize is None : 
            markersize=mpl.rcParams['lines.markersize']
        if dash_capstyle is None : 
            dash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if dash_joinstyle is None : 
            dash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if solid_capstyle is None : 
            solid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if solid_joinstyle is None : 
            solid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']
        
        # Create the alpha range        
        r = np.arange(mindegrees,maxdegrees+degreestep,degreestep)
        if any(r>90.0) or any(r<0.0):
            print 'valid range for grid_rings is between 0 and 90'

        # Put data in our valid range
        r = r * np.pi/180.0            
        r = r[r>0.0]
        r = r[r<np.pi/2.0]
        r = r[r!=np.pi/2.0]
        th = np.arange(0.0,2.0*np.pi+np.pi/resolution,2.0*np.pi/resolution)

        for alpha in r:
            radius=np.sin(alpha)/(1.0+np.cos(alpha))
            plt.plot(radius*np.cos(th),radius*np.sin(th),
                     linewidth=linewidth,
                     linestyle=linestyle,
                     color=color,
                     marker=marker,
                     markersize=markersize,
                     markeredgewidth=markeredgewidth,
                     markeredgecolor=markeredgecolor,
                     markerfacecolor=markerfacecolor,
                     markerfacecoloralt=markerfacecoloralt,
                     fillstyle=fillstyle,
                     dash_capstyle=dash_capstyle,
                     solid_capstyle=solid_capstyle,
                     dash_joinstyle=dash_joinstyle,
                     solid_joinstyle=solid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     zorder=zorder,#place grid b/t background and data points
                     **kwargs)

                
            plt.axes(plt.gca()).set_xbound(-1.05,1.05)
            plt.axes(plt.gca()).set_ybound(-1.05,1.05)
            
#----------------------------------------------------------------------------

    def grid_wulffnet(self,degreestep1=10.0, 
                             degreestep2=2.0,
                             rotation=0.0,
                             poleclip1=2.0,
                             poleclip2=10.0,
                             resolution=25,
                             linewidth1=2,
                             linewidth2=1,
                             linestyle='-',
                             color=[0.6,0.6,0.6],
                             marker=None,
                             markersize=None,
                             markeredgewidth='none',
                             markeredgecolor='none',
                             markerfacecolor='none',
                             markerfacecoloralt='none',
                             fillstyle='full',
                             dash_capstyle=None,
                             solid_capstyle=None,
                             dash_joinstyle=None,
                             solid_joinstyle=None,
                             pickradius=None,
                             drawstyle=None,
                             markevery=None,
                             antialiased=None,
                             **kwargs):

        plt.subplot(self._subplot)
                                     
        self.grid_greatcircles(degreestep=degreestep1,
                                 mindegrees=0.0,maxdegrees=180.0,
                                 resolution=resolution,
                                 linewidth=linewidth1,
                                 linestyle=linestyle,
                                 color=color,
                                 poleclip=poleclip1,
                                 rotation=rotation,
                                 marker=marker,
                                 markersize=markersize,
                                 markeredgewidth=markeredgewidth,
                                 markeredgecolor=markeredgecolor,
                                 markerfacecolor=markerfacecolor,
                                 markerfacecoloralt=markerfacecoloralt,
                                 fillstyle=fillstyle,
                                 dash_capstyle=dash_capstyle,
                                 solid_capstyle=solid_capstyle,
                                 dash_joinstyle=dash_joinstyle,
                                 solid_joinstyle=solid_joinstyle,
                                 pickradius=pickradius,
                                 drawstyle=drawstyle,
                                 markevery=markevery,
                                 antialiased=antialiased,
                                 **kwargs)

        self.grid_greatcircles(degreestep=degreestep2,
                                 mindegrees=0.0,maxdegrees=180.0,
                                 resolution=resolution,
                                 linewidth=linewidth2,
                                 linestyle=linestyle,
                                 color=color,
                                 poleclip=poleclip2,
                                 rotation=rotation,
                                 marker=marker,
                                 markersize=markersize,
                                 markeredgewidth=markeredgewidth,
                                 markeredgecolor=markeredgecolor,
                                 markerfacecolor=markerfacecolor,
                                 markerfacecoloralt=markerfacecoloralt,
                                 fillstyle=fillstyle,
                                 dash_capstyle=dash_capstyle,
                                 solid_capstyle=solid_capstyle,
                                 dash_joinstyle=dash_joinstyle,
                                 solid_joinstyle=solid_joinstyle,
                                 pickradius=pickradius,
                                 drawstyle=drawstyle,
                                 markevery=markevery,
                                 antialiased=antialiased,
                                 **kwargs)


        self.grid_smallcircles(degreestep=degreestep1,
                                 mindegrees=0.0,maxdegrees=180.0,
                                 resolution=resolution,
                                 linewidth=linewidth1,
                                 linestyle=linestyle,
                                 color=color,
                                 rotation=rotation,
                                 marker=marker,
                                 markersize=markersize,
                                 markeredgewidth=markeredgewidth,
                                 markeredgecolor=markeredgecolor,
                                 markerfacecolor=markerfacecolor,
                                 markerfacecoloralt=markerfacecoloralt,
                                 fillstyle=fillstyle,
                                 dash_capstyle=dash_capstyle,
                                 solid_capstyle=solid_capstyle,
                                 dash_joinstyle=dash_joinstyle,
                                 solid_joinstyle=solid_joinstyle,
                                 pickradius=pickradius,
                                 drawstyle=drawstyle,
                                 markevery=markevery,
                                 antialiased=antialiased,
                                 **kwargs)

        self.grid_smallcircles(degreestep=degreestep2,
                                 mindegrees=0.0,maxdegrees=180.0,
                                 resolution=resolution,
                                 linewidth=linewidth2,
                                 linestyle=linestyle,
                                 color=color,
                                 rotation=rotation,
                                 marker=marker,
                                 markersize=markersize,
                                 markeredgewidth=markeredgewidth,
                                 markeredgecolor=markeredgecolor,
                                 markerfacecolor=markerfacecolor,
                                 markerfacecoloralt=markerfacecoloralt,
                                 fillstyle=fillstyle,
                                 dash_capstyle=dash_capstyle,
                                 solid_capstyle=solid_capstyle,
                                 dash_joinstyle=dash_joinstyle,
                                 solid_joinstyle=solid_joinstyle,
                                 pickradius=pickradius,
                                 drawstyle=drawstyle,
                                 markevery=markevery,
                                 antialiased=antialiased,
                                 **kwargs)

        self.grid_spokes(degreestep=90.0,mindegrees=0.0,maxdegrees=360.0,
                     resolution=resolution,
                     rotation=rotation,
                     linewidth=linewidth1,
                     linestyle=linestyle,
                     color=color,
                     marker=marker,
                     markersize=markersize,
                     markeredgewidth=markeredgewidth,
                     markeredgecolor=markeredgecolor,
                     markerfacecolor=markerfacecolor,
                     markerfacecoloralt=markerfacecoloralt,
                     fillstyle=fillstyle,
                     dash_capstyle=dash_capstyle,
                     solid_capstyle=solid_capstyle,
                     dash_joinstyle=dash_joinstyle,
                     solid_joinstyle=solid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)

#----------------------------------------------------------------------------

    def add_eastlabel(self,x=1.05, y=0.0, maxval=9, 
                      verticalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        rat=rationalize(self._east,maxval=maxval)
        txt=str(int(rat[0]))+' '+str(int(rat[1]))+' '+str(int(rat[2]))
        plt.text(x,y,txt,verticalalignment=verticalalignment,**kwargs)
        
        plt.axes(plt.gca()).set_xbound(-1.15,1.15)
        plt.axes(plt.gca()).set_ybound(-1.15,1.15)        

#----------------------------------------------------------------------------

    def add_southlabel(self,x=0.00, y=-1.1, maxval=9, 
                      horizontalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        rat=rationalize(self._south,maxval=maxval)
        txt=str(int(rat[0]))+' '+str(int(rat[1]))+' '+str(int(rat[2]))
        plt.text(x,y,txt,horizontalalignment=horizontalalignment,**kwargs)

        plt.axes(plt.gca()).set_xbound(-1.15,1.15)
        plt.axes(plt.gca()).set_ybound(-1.15,1.15)  

#----------------------------------------------------------------------------

    def add_centerlabel(self,x=-0.90, y=-0.90, maxval=9, 
                      horizontalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        rat=rationalize(self._center,maxval=maxval)
        txt=str(int(rat[0]))+' '+str(int(rat[1]))+' '+str(int(rat[2]))+'\n' \
            +'Pole Figure'
        plt.text(x,y,txt,horizontalalignment=horizontalalignment,**kwargs)

        plt.axes(plt.gca()).set_xbound(-1.15,1.15)
        plt.axes(plt.gca()).set_ybound(-1.15,1.15)  

#----------------------------------------------------------------------------

    def add_millerlabels(self,poles,uc,xoffset=0.05,yoffset=0.05,
                       maxval=9,horizontalalignment='left',
                       verticalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        if isinstance(poles,miller):
            
            x,y,hem = stereotrans(poles.to_cartesian(uc),center=self._center,
                        east=self._east,south=self._south)
            
            i=-1
            for item in poles:
                i=i+1
                rat=rationalize([poles[i].h,poles[i].k,poles[i].l],
                                maxval=maxval)
                txt='('+str(int(rat[0]))+' '+str(int(rat[1]))+' ' \
                    +str(int(rat[2]))+')'
                plt.text(x[i]+xoffset,y[i]+yoffset,txt,
                     horizontalalignment=horizontalalignment,**kwargs)
                
        else:
            print('PoleFigure method add_millerlabels requires \
                   miller class as input')

#----------------------------------------------------------------------------

    def add_lattveclabels(self,vects,uc,xoffset=0.05,yoffset=-0.05,
                       maxval=9,horizontalalignment='left',
                       verticalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        if isinstance(vects,lattvec):
            
            x,y,hem = stereotrans(vects.to_cartesian(uc), center=self._center,
                        east=self._east, south=self._south)
            
            i=-1
            for item in vects:
                i=i+1
                rat=rationalize([vects[i].u, vects[i].v, vects[i].w],
                                maxval=maxval)
                txt='['+str(int(rat[0]))+' '+str(int(rat[1]))+' ' \
                    +str(int(rat[2]))+']'
                plt.text(x[i]+xoffset,y[i]+yoffset,txt,
                     horizontalalignment=horizontalalignment,**kwargs)
        else:
            print('PoleFigure method add_lattveclabels requires \
                   lattvec class as input')

#----------------------------------------------------------------------------

    def add_coordinatereadout(self,uc):

        plt.subplot(self._subplot)     
        def format_coord(x, y):
            """
            Override this method to change how the values are displayed in
            the status bar.
        
            In this case, we want them to be displayed in degrees N/S/E/W.
            """
            
            xyz=revstereotrans([x,y,'N'])
            uvw,dev=lattvec.from_cartesian(xyz,uc,maxval=29)
            uvw=np.int8(np.squeeze(np.vstack([uvw.u,uvw.v,uvw.w])))
            hkl,dev=miller.from_cartesian(xyz,uc,maxval=29)
            hkl=np.int8(np.squeeze(np.vstack([hkl.h,hkl.k,hkl.l])))
            stl='{:>3}'
            uvwn=stl.format(uvw[0])+' '+stl.format(uvw[1])+' ' \
                 +stl.format(uvw[2])
            hkln=stl.format(hkl[0])+' '+stl.format(hkl[1])+' ' \
                 +stl.format(hkl[2])
            
            xyz=revstereotrans([x,y,'S'])
            uvw,dev=lattvec.from_cartesian(xyz,uc,maxval=29)
            uvw=np.int8(np.squeeze(np.vstack([uvw.u,uvw.v,uvw.w])))
            hkl,dev=miller.from_cartesian(xyz,uc,maxval=29)
            hkl=np.int8(np.squeeze(np.vstack([hkl.h,hkl.k,hkl.l])))
            uvws=stl.format(uvw[0])+' '+stl.format(uvw[1])+' ' \
                 +stl.format(uvw[2])
            hkls=stl.format(hkl[0])+' '+stl.format(hkl[1])+' ' \
                 +stl.format(hkl[2])
            
            output='Upper: ['+uvwn+'] /'+ \
                   ' ('+hkln+')'+ \
                   '      '+ \
                   'Lower: ['+uvws+'] /'+ \
                   ' ('+hkls+')'
        
            return output
        
        self._ax.format_coord=format_coord

#----------------------------------------------------------------------------

    def grid_standard(self,uc,resolution=25,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=1,
                    upperlinestyle='-',
                    uppercolor=[0.6,0.6,0.6],
                    uppermarker=None,
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='none',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=1,
                    lowerlinestyle='-',
                    lowercolor=[0.6,0.6,0.6],
                    lowermarker=None,
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='none',
                    lowermarkerfacecolor='none',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                    
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    zorder=1,
                    **kwargs):

        plt.subplot(self._subplot)
        
        h = [ 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0]
        k = [-1, 1,-1, 1, 0, 0, 0, 0, 1, 1,-1,-1, 0, 0, 1,-1, 0, 0]
        l = [ 0, 0, 0, 0,-1, 1,-1, 1,-1, 1,-1, 1, 0, 0, 0, 0, 1,-1]
        
        self.add_trace(miller(h,k,l), uc,
                    upperlinewidth=upperlinewidth,
                    upperlinestyle=upperlinestyle,
                    uppercolor=uppercolor,
                    uppermarker=uppermarker,
                    uppermarkersize=uppermarkersize,
                    uppermarkeredgewidth=uppermarkeredgewidth,
                    uppermarkeredgecolor=uppermarkeredgecolor,
                    uppermarkerfacecolor=uppermarkerfacecolor,
                    uppermarkerfacecoloralt=uppermarkerfacecoloralt,
                    upperfillstyle=upperfillstyle,
                    upperdash_capstyle=upperdash_capstyle,
                    uppersolid_capstyle=uppersolid_capstyle,
                    upperdash_joinstyle=upperdash_joinstyle,
                    uppersolid_joinstyle=uppersolid_joinstyle,
                    lowerlinewidth=lowerlinewidth,
                    lowerlinestyle=lowerlinestyle,
                    lowercolor=lowercolor,
                    lowermarker=lowermarker,
                    lowermarkersize=lowermarkersize,
                    lowermarkeredgewidth=lowermarkeredgewidth,
                    lowermarkeredgecolor=lowermarkeredgecolor,
                    lowermarkerfacecolor=lowermarkerfacecolor,
                    lowermarkerfacecoloralt=lowermarkerfacecoloralt,
                    lowerfillstyle=lowerfillstyle,
                    lowerdash_capstyle=lowerdash_capstyle,
                    lowersolid_capstyle=lowersolid_capstyle,
                    lowerdash_joinstyle=lowerdash_joinstyle,
                    lowersolid_joinstyle=lowersolid_joinstyle,
                    pickradius=pickradius,
                    drawstyle=drawstyle,
                    markevery=markevery,
                    antialiased=antialiased,
                    zorder=zorder,
                    **kwargs)



#----------------------------------------------------------------------------

class eaproj(object):
    '''
    A stereogram represents the planes and the directions of a crystal.
    A pole figure is a stereogram of the planes in the figure.
    '''    
    
    
    def __init__(self, figure=None, subplot=None, 
                 center=[0,0,1], south=[0,1,0], east=[1,0,0], **kwargs):
        
        if not 'figsize' in kwargs:
            kwargs['figsize'] = (10, 10)
        
        # Open the figure
        if figure:
            self._figure = figure
        else:
            self._figure = plt.figure(**kwargs)
            
        # Create first axis instance
        if subplot:
            self._ax = self._figure.add_subplot(subplot)
            self._subplot = subplot
        else:
            self._ax = self._figure.add_subplot(111)
            self._subplot = 111
        
        # Create the background
        circ=plt.Circle((0,0),1,facecolor='w',edgecolor='k',linewidth=1,alpha=1)
        self._ax.add_patch(circ)
        plt.axes(self._ax).set_xbound(-1.05,1.05)
        plt.axes(self._ax).set_ybound(-1.05,1.05)
        plt.axes(self._ax).set_aspect('equal')
        plt.axes(self._ax).axison = False
    
        south = vecarrayconvert(south)
        east = vecarrayconvert(east)
        center = vecarrayconvert(center)
                
        if  abs(vecarraydot(south,east))<np.sqrt(np.spacing(1)) and \
            abs(vecarraydot(east,center))<np.sqrt(np.spacing(1)) and \
            abs(vecarraydot(center,south))<np.sqrt(np.spacing(1)):
                      
            self._center=center
            self._south=south
            self._east=east
        else:
            print('The pole figure center, south, and east directions must be \
                orthonormal. \
                Proceeding with a standard projection')
            self._center=[0,0,1]
            self._south=[0,1,0]
            self._east=[1,0,0]

        if  vecarraydot(vecarraycross(south,east),center)<np.sqrt(np.spacing(1)):
                      
            self._center=center
            self._south=south
            self._east=east
        else:
            print('The pole figure center, south, and east directions must \
                form a right-handed system. \
                Proceeding with a standard projection')
            self._center=[0,0,1]
            self._south=[0,1,0]
            self._east=[1,0,0]

#-------------------------------------------------------------------------------

    # Method for adding directions to the pole figure
    def add_lattvec(self,v,uc,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=None,
                    upperlinestyle='none',
                    uppercolor='r',
                    uppermarker='o',
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='r',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=None,
                    lowerlinestyle='none',
                    lowercolor='none',
                    lowermarker='o',
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='r',
                    lowermarkerfacecolor='r',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                     
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    **kwargs):
        
        # assign relevant defaults from mpl.rcParams
        if upperlinewidth is None : 
            upperlinewidth=mpl.rcParams['lines.linewidth']
        if upperlinestyle is None : 
            upperlinestyle=mpl.rcParams['lines.linestyle']
        if uppermarker is None : 
            uppermarker=mpl.rcParams['lines.marker']
        if uppercolor is None : 
            uppercolor=mpl.rcParams['lines.color']
        if uppermarkersize is None : 
            uppermarkersize=mpl.rcParams['lines.markersize']
        if upperdash_capstyle is None : 
            upperdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if upperdash_joinstyle is None : 
            upperdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if uppersolid_capstyle is None : 
            uppersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if uppersolid_joinstyle is None : 
            uppersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if lowerlinewidth is None : 
            lowerlinewidth=mpl.rcParams['lines.linewidth']
        if lowerlinestyle is None : 
            lowerlinestyle=mpl.rcParams['lines.linestyle']
        if lowermarker is None : 
            lowermarker=mpl.rcParams['lines.marker']
        if lowercolor is None : 
            lowercolor=mpl.rcParams['lines.color']
        if lowermarkersize is None : 
            lowermarkersize=mpl.rcParams['lines.markersize']
        if lowerdash_capstyle is None : 
            lowerdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if lowerdash_joinstyle is None : 
            lowerdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if lowersolid_capstyle is None : 
            lowersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if lowersolid_joinstyle is None : 
            lowersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']
        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']

        if isinstance(v,lattvec):
            
            plt.subplot(self._subplot)
            
            x,y,hem = eatrans(v.to_cartesian(uc))
            
            plt.plot(x[hem=='N'],y[hem=='N'],
                     linewidth=upperlinewidth,
                     linestyle=upperlinestyle,
                     color=uppercolor,
                     marker=uppermarker,
                     markersize=uppermarkersize,
                     markeredgewidth=uppermarkeredgewidth,
                     markeredgecolor=uppermarkeredgecolor,
                     markerfacecolor=uppermarkerfacecolor,
                     markerfacecoloralt=uppermarkerfacecoloralt,
                     fillstyle=upperfillstyle,
                     dash_capstyle=upperdash_capstyle,
                     solid_capstyle=uppersolid_capstyle,
                     dash_joinstyle=upperdash_joinstyle,
                     solid_joinstyle=uppersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)
                     
            plt.plot(x[hem=='S'],y[hem=='S'],                     
                     linewidth=lowerlinewidth,
                     linestyle=lowerlinestyle,
                     color=lowercolor,
                     marker=lowermarker,
                     markersize=lowermarkersize,
                     markeredgewidth=lowermarkeredgewidth,
                     markeredgecolor=lowermarkeredgecolor,
                     markerfacecolor=lowermarkerfacecolor,
                     markerfacecoloralt=lowermarkerfacecoloralt,
                     fillstyle=lowerfillstyle,
                     dash_capstyle=lowerdash_capstyle,
                     solid_capstyle=lowersolid_capstyle,
                     dash_joinstyle=lowerdash_joinstyle,
                     solid_joinstyle=lowersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)

            plt.axes(self._ax).set_xbound(-1.05,1.05)
            plt.axes(self._ax).set_ybound(-1.05,1.05)

#----------------------------------------------------------------------------

    # Method for adding planes to the pole figure
    def add_miller(self,v,uc,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=None,
                    upperlinestyle='none',
                    uppercolor='b',
                    uppermarker='s',
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='b',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=None,
                    lowerlinestyle='none',
                    lowercolor='none',
                    lowermarker='s',
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='b',
                    lowermarkerfacecolor='b',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                    
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    **kwargs):

        plt.subplot(self._subplot)
        
        # assign relevant defaults from mpl.rcParams
        if upperlinewidth is None : 
            upperlinewidth=mpl.rcParams['lines.linewidth']
        if upperlinestyle is None : 
            upperlinestyle=mpl.rcParams['lines.linestyle']
        if uppermarker is None : 
            uppermarker=mpl.rcParams['lines.marker']
        if uppercolor is None : 
            uppercolor=mpl.rcParams['lines.color']
        if uppermarkersize is None : 
            uppermarkersize=mpl.rcParams['lines.markersize']
        if upperdash_capstyle is None : 
            upperdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if upperdash_joinstyle is None : 
            upperdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if uppersolid_capstyle is None : 
            uppersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if uppersolid_joinstyle is None : 
            uppersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if lowerlinewidth is None : 
            lowerlinewidth=mpl.rcParams['lines.linewidth']
        if lowerlinestyle is None : 
            lowerlinestyle=mpl.rcParams['lines.linestyle']
        if lowermarker is None : 
            lowermarker=mpl.rcParams['lines.marker']
        if lowercolor is None : 
            lowercolor=mpl.rcParams['lines.color']
        if lowermarkersize is None : 
            lowermarkersize=mpl.rcParams['lines.markersize']
        if lowerdash_capstyle is None : 
            lowerdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if lowerdash_joinstyle is None : 
            lowerdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if lowersolid_capstyle is None : 
            lowersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if lowersolid_joinstyle is None : 
            lowersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']
        
        if isinstance(v,miller):
            
            x,y,hem = eatrans(v.to_cartesian(uc), center=self._center,
                              east=self._east, south=self._south)
            
            plt.plot(x[hem=='N'],y[hem=='N'],
                     linewidth=upperlinewidth,
                     linestyle=upperlinestyle,
                     color=uppercolor,
                     marker=uppermarker,
                     markersize=uppermarkersize,
                     markeredgewidth=uppermarkeredgewidth,
                     markeredgecolor=uppermarkeredgecolor,
                     markerfacecolor=uppermarkerfacecolor,
                     markerfacecoloralt=uppermarkerfacecoloralt,
                     fillstyle=upperfillstyle,
                     dash_capstyle=upperdash_capstyle,
                     solid_capstyle=uppersolid_capstyle,
                     dash_joinstyle=upperdash_joinstyle,
                     solid_joinstyle=uppersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)
                     
            plt.plot(x[hem=='S'],y[hem=='S'],                     
                     linewidth=lowerlinewidth,
                     linestyle=lowerlinestyle,
                     color=lowercolor,
                     marker=lowermarker,
                     markersize=lowermarkersize,
                     markeredgewidth=lowermarkeredgewidth,
                     markeredgecolor=lowermarkeredgecolor,
                     markerfacecolor=lowermarkerfacecolor,
                     markerfacecoloralt=lowermarkerfacecoloralt,
                     fillstyle=lowerfillstyle,
                     dash_capstyle=lowerdash_capstyle,
                     solid_capstyle=lowersolid_capstyle,
                     dash_joinstyle=lowerdash_joinstyle,
                     solid_joinstyle=lowersolid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     **kwargs)

        else:
            print 'add_miller method for pole figures \
                   only accepts miller class input'

#----------------------------------------------------------------------------

    def add_trace(self,v,uc,
                    resolution=25,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=1,
                    upperlinestyle='-',
                    uppercolor='b',
                    uppermarker=None,
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='none',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=1,
                    lowerlinestyle='--',
                    lowercolor='b',
                    lowermarker=None,
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='none',
                    lowermarkerfacecolor='none',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                    
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    zorder=2,
                    **kwargs):

        plt.subplot(self._subplot)
               
        # assign relevant defaults from mpl.rcParams
        if upperlinewidth is None : 
            upperlinewidth=mpl.rcParams['lines.linewidth']
        if upperlinestyle is None : 
            upperlinestyle=mpl.rcParams['lines.linestyle']
        if uppermarker is None : 
            uppermarker=mpl.rcParams['lines.marker']
        if uppercolor is None : 
            uppercolor=mpl.rcParams['lines.color']
        if uppermarkersize is None : 
            uppermarkersize=mpl.rcParams['lines.markersize']
        if upperdash_capstyle is None : 
            upperdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if upperdash_joinstyle is None : 
            upperdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if uppersolid_capstyle is None : 
            uppersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if uppersolid_joinstyle is None : 
            uppersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if lowerlinewidth is None : 
            lowerlinewidth=mpl.rcParams['lines.linewidth']
        if lowerlinestyle is None : 
            lowerlinestyle=mpl.rcParams['lines.linestyle']
        if lowermarker is None : 
            lowermarker=mpl.rcParams['lines.marker']
        if lowercolor is None : 
            lowercolor=mpl.rcParams['lines.color']
        if lowermarkersize is None : 
            lowermarkersize=mpl.rcParams['lines.markersize']
        if lowerdash_capstyle is None : 
            lowerdash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if lowerdash_joinstyle is None : 
            lowerdash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if lowersolid_capstyle is None : 
            lowersolid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if lowersolid_joinstyle is None : 
            lowersolid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']
        
        if isinstance(v,miller):
            
            abc = vecarrayconvert(self._center)
            nrmabc = 1.0/vecarraynorm(abc)    
            
            uvw = vecarrayconvert(self._east)
            nrmuvw = 1.0/vecarraynorm(uvw)
            
            fed = vecarrayconvert(self._south) # 'def' is reserved in python
            nrmdef = 1.0/vecarraynorm(fed) 
            
            xyz = v.to_cartesian(uc)
            nrmxyz = 1.0/vecarraynorm(xyz)
            
            xx = xyz[0]*nrmxyz
            yy = xyz[1]*nrmxyz
            zz = xyz[2]*nrmxyz
        
            n = np.shape(xx)     
            
            aa = np.tile(abc[0]*nrmabc,n)
            bb = np.tile(abc[1]*nrmabc,n)
            cc = np.tile(abc[2]*nrmabc,n)
            dd = np.tile(fed[0]*nrmdef,n)
            ee = np.tile(fed[1]*nrmdef,n)
            ff = np.tile(fed[2]*nrmdef,n)
            uu = np.tile(uvw[0]*nrmuvw,n)
            vv = np.tile(uvw[1]*nrmuvw,n)
            ww = np.tile(uvw[2]*nrmuvw,n)
        
            cosdl = vecarraydot([xx,yy,zz],[dd,ee,ff])
            cosmu = vecarraydot([xx,yy,zz],[uu,vv,ww])
            cosal = vecarraydot([xx,yy,zz],[aa,bb,cc])
            
            denom = 1.0/np.sqrt(1.0+abs(cosal))
            
            xproj = cosmu * denom
            yproj = cosdl * denom
        
            hemis = np.tile('N',n)    
            if np.array(n)>1:
                hemis[cosal<0.0] = 'S'
            else:
                if cosal<0.0:
                    hemis = np.tile('S',n)
            
            denom = np.sqrt(xproj*xproj+yproj*yproj)
            denom[denom==0.0]=1.0
            cosrho = xproj/denom
            sinrho = yproj/denom
            cosrho[denom==0.0]=0.0
            sinrho[denom==0.0]=0.0
            
            alpha = np.arccos(cosal)
    
    
            i=-1
            for almx in alpha:
                
                # 1/cos and tan have problems near pi/2
                if (abs(almx)-np.pi/2.0)<np.sqrt(np.spacing(1)):
                    almx=abs(almx)-np.sqrt(np.spacing(1))
                    
                
                rgc = 1.0/np.cos(almx)
                ogc = np.tan(almx)
                
                thi = np.pi/2.0+almx
                thf = 3.0*np.pi/2.0-almx
                ths = (thf-thi)/resolution
    
                th  = np.arange(thi,thf+ths/2.0,ths)
                x = rgc*np.cos(th)+ogc
                y = rgc*np.sin(th)

                i=i+1                
                cr = cosrho[i]
                sr = sinrho[i]

                if hemis[i]=='N':
                    plt.plot(x*cr-y*sr,y*cr+x*sr,
                             linewidth=upperlinewidth,
                             linestyle=upperlinestyle,
                             color=uppercolor,
                             marker=uppermarker,
                             markersize=uppermarkersize,
                             markeredgewidth=uppermarkeredgewidth,
                             markeredgecolor=uppermarkeredgecolor,
                             markerfacecolor=uppermarkerfacecolor,
                             markerfacecoloralt=uppermarkerfacecoloralt,
                             fillstyle=upperfillstyle,
                             dash_capstyle=upperdash_capstyle,
                             solid_capstyle=uppersolid_capstyle,
                             dash_joinstyle=upperdash_joinstyle,
                             solid_joinstyle=uppersolid_joinstyle,
                             pickradius=pickradius,
                             drawstyle=drawstyle,
                             markevery=markevery,
                             antialiased=antialiased,
                             zorder=zorder,
                             **kwargs)
                             
                if hemis[i]=='S':
                    plt.plot(-x*cr+y*sr,-y*cr-x*sr,
                             linewidth=lowerlinewidth,
                             linestyle=lowerlinestyle,
                             color=lowercolor,
                             marker=lowermarker,
                             markersize=lowermarkersize,
                             markeredgewidth=lowermarkeredgewidth,
                             markeredgecolor=lowermarkeredgecolor,
                             markerfacecolor=lowermarkerfacecolor,
                             markerfacecoloralt=lowermarkerfacecoloralt,
                             fillstyle=lowerfillstyle,
                             dash_capstyle=lowerdash_capstyle,
                             solid_capstyle=lowersolid_capstyle,
                             dash_joinstyle=lowerdash_joinstyle,
                             solid_joinstyle=lowersolid_joinstyle,
                             pickradius=pickradius,
                             drawstyle=drawstyle,
                             markevery=markevery,
                             antialiased=antialiased,
                             zorder=zorder,
                             **kwargs)                             

#----------------------------------------------------------------------------

    def grid_spokes(self,degreestep=90.0,mindegrees=0.0,maxdegrees=360.0,
                     resolution=25,
                     poleclip=0,
                     rotation=0.0,
                     linewidth=1,
                     linestyle='-',
                     color=[0.6,0.6,0.6],
                     marker=None,
                     markersize=None,
                     markeredgewidth='none',
                     markeredgecolor='none',
                     markerfacecolor='none',
                     markerfacecoloralt='none',
                     fillstyle='full',
                     dash_capstyle=None,
                     solid_capstyle=None,
                     dash_joinstyle=None,
                     solid_joinstyle=None,
                     pickradius=None,
                     drawstyle=None,
                     markevery=None,
                     antialiased=None,
                     zorder=1,
                     **kwargs):

        plt.subplot(self._subplot)

        # assign relevant defaults from mpl.rcParams
        if linewidth is None : 
            linewidth=mpl.rcParams['lines.linewidth']
        if linestyle is None : 
            linestyle=mpl.rcParams['lines.linestyle']
        if marker is None : 
            marker=mpl.rcParams['lines.marker']
        if color is None : 
            color=mpl.rcParams['lines.color']
        if markersize is None : 
            markersize=mpl.rcParams['lines.markersize']
        if dash_capstyle is None : 
            dash_capstyle=mpl.rcParams['lines.dash_capstyle']
        if dash_joinstyle is None : 
            dash_joinstyle=mpl.rcParams['lines.dash_joinstyle']
        if solid_capstyle is None : 
            solid_capstyle=mpl.rcParams['lines.solid_capstyle']
        if solid_joinstyle is None : 
            solid_joinstyle=mpl.rcParams['lines.solid_joinstyle']

        if drawstyle is None : 
            drawstyle='default'
        if antialiased is None : 
            antialiased=mpl.rcParams['lines.antialiased']

        poleclip=radians(poleclip)
        
        th = np.arange(mindegrees,maxdegrees+degreestep,degreestep)
        th = th * np.pi / 180.0
        lin = np.arange(poleclip,1.0+0.5/resolution,1.0/resolution)
        
        rho = radians(rotation)
        cosrho = np.cos(rho)
        sinrho = np.sin(rho)
        

        
        for alph in th:
            
            x = lin * np.cos(alph)
            y = lin * np.sin(alph)
              
            plt.plot(x*cosrho-y*sinrho,y*cosrho+x*sinrho,
                     linewidth=linewidth,
                     linestyle=linestyle,
                     color=color,
                     marker=marker,
                     markersize=markersize,
                     markeredgewidth=markeredgewidth,
                     markeredgecolor=markeredgecolor,
                     markerfacecolor=markerfacecolor,
                     markerfacecoloralt=markerfacecoloralt,
                     fillstyle=fillstyle,
                     dash_capstyle=dash_capstyle,
                     solid_capstyle=solid_capstyle,
                     dash_joinstyle=dash_joinstyle,
                     solid_joinstyle=solid_joinstyle,
                     pickradius=pickradius,
                     drawstyle=drawstyle,
                     markevery=markevery,
                     antialiased=antialiased,
                     zorder=zorder,
                     **kwargs)
            
            plt.axes(plt.gca()).set_xbound(-1.05,1.05)
            plt.axes(plt.gca()).set_ybound(-1.05,1.05)

#----------------------------------------------------------------------------

    def add_eastlabel(self,x=1.05, y=0.0, maxval=9, 
                      verticalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        rat=rationalize(self._east,maxval=maxval)
        txt=str(int(rat[0]))+' '+str(int(rat[1]))+' '+str(int(rat[2]))
        plt.text(x,y,txt,verticalalignment=verticalalignment,**kwargs)
        
        plt.axes(plt.gca()).set_xbound(-1.15,1.15)
        plt.axes(plt.gca()).set_ybound(-1.15,1.15)        

#----------------------------------------------------------------------------

    def add_southlabel(self,x=0.00, y=-1.1, maxval=9, 
                      horizontalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        rat=rationalize(self._south,maxval=maxval)
        txt=str(int(rat[0]))+' '+str(int(rat[1]))+' '+str(int(rat[2]))
        plt.text(x,y,txt,horizontalalignment=horizontalalignment,**kwargs)

        plt.axes(plt.gca()).set_xbound(-1.15,1.15)
        plt.axes(plt.gca()).set_ybound(-1.15,1.15)  

#----------------------------------------------------------------------------

    def add_centerlabel(self,x=-0.90, y=-0.90, maxval=9, 
                      horizontalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        rat=rationalize(self._center,maxval=maxval)
        txt=str(int(rat[0]))+' '+str(int(rat[1]))+' '+str(int(rat[2]))+'\n' \
            +'Pole Figure'
        plt.text(x,y,txt,horizontalalignment=horizontalalignment,**kwargs)

        plt.axes(plt.gca()).set_xbound(-1.15,1.15)
        plt.axes(plt.gca()).set_ybound(-1.15,1.15)  

#----------------------------------------------------------------------------

    def add_millerlabels(self,poles,uc,xoffset=0.05,yoffset=0.05,
                       maxval=9,horizontalalignment='left',
                       verticalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        if isinstance(poles,miller):
            
            x,y,hem = eatrans(poles.to_cartesian(uc),center=self._center,
                        east=self._east,south=self._south)
            
            i=-1
            for item in poles:
                i=i+1
                rat=rationalize([poles[i].h,poles[i].k,poles[i].l],
                                maxval=maxval)
                txt='('+str(int(rat[0]))+' '+str(int(rat[1]))+' ' \
                    +str(int(rat[2]))+')'
                plt.text(x[i]+xoffset,y[i]+yoffset,txt,
                     horizontalalignment=horizontalalignment,**kwargs)
                
        else:
            print('PoleFigure method add_millerlabels requires \
                   miller class as input')

#----------------------------------------------------------------------------

    def add_lattveclabels(self,vects,uc,xoffset=0.05,yoffset=-0.05,
                       maxval=9,horizontalalignment='left',
                       verticalalignment='center',**kwargs):

        plt.subplot(self._subplot)
        if isinstance(vects,lattvec):
            
            x,y,hem = eatrans(vects.to_cartesian(uc), center=self._center,
                        east=self._east, south=self._south)
            
            i=-1
            for item in vects:
                i=i+1
                rat=rationalize([vects[i].u, vects[i].v, vects[i].w],
                                maxval=maxval)
                txt='['+str(int(rat[0]))+' '+str(int(rat[1]))+' ' \
                    +str(int(rat[2]))+']'
                plt.text(x[i]+xoffset,y[i]+yoffset,txt,
                     horizontalalignment=horizontalalignment,**kwargs)
        else:
            print('PoleFigure method add_lattveclabels requires \
                   lattvec class as input')

#----------------------------------------------------------------------------

    def add_coordinatereadout(self,uc):

        plt.subplot(self._subplot)     
        def format_coord(x, y):
            """
            Override this method to change how the values are displayed in
            the status bar.
        
            In this case, we want them to be displayed in degrees N/S/E/W.
            """
            
            xyz=reveatrans([x,y,'N'])
            uvw,dev=lattvec.from_cartesian(xyz,uc,maxval=29)
            uvw=np.int8(np.squeeze(np.vstack([uvw.u,uvw.v,uvw.w])))
            hkl,dev=miller.from_cartesian(xyz,uc,maxval=29)
            hkl=np.int8(np.squeeze(np.vstack([hkl.h,hkl.k,hkl.l])))
            stl='{:>3}'
            uvwn=stl.format(uvw[0])+' '+stl.format(uvw[1])+' ' \
                 +stl.format(uvw[2])
            hkln=stl.format(hkl[0])+' '+stl.format(hkl[1])+' ' \
                 +stl.format(hkl[2])
            
            xyz=reveatrans([x,y,'S'])
            uvw,dev=lattvec.from_cartesian(xyz,uc,maxval=29)
            uvw=np.int8(np.squeeze(np.vstack([uvw.u,uvw.v,uvw.w])))
            hkl,dev=miller.from_cartesian(xyz,uc,maxval=29)
            hkl=np.int8(np.squeeze(np.vstack([hkl.h,hkl.k,hkl.l])))
            uvws=stl.format(uvw[0])+' '+stl.format(uvw[1])+' ' \
                 +stl.format(uvw[2])
            hkls=stl.format(hkl[0])+' '+stl.format(hkl[1])+' ' \
                 +stl.format(hkl[2])
            
            output='Upper: ['+uvwn+'] /'+ \
                   ' ('+hkln+')'+ \
                   '      '+ \
                   'Lower: ['+uvws+'] /'+ \
                   ' ('+hkls+')'
        
            return output
        
        self._ax.format_coord=format_coord

#----------------------------------------------------------------------------

    def grid_standard(self,uc,resolution=25,
                    
                    # upper hemisphere options & defaults
                    upperlinewidth=1,
                    upperlinestyle='-',
                    uppercolor=[0.6,0.6,0.6],
                    uppermarker=None,
                    uppermarkersize=None,
                    uppermarkeredgewidth=None,
                    uppermarkeredgecolor='none',
                    uppermarkerfacecolor='none',
                    uppermarkerfacecoloralt='none',
                    upperfillstyle='full',
                    upperdash_capstyle=None,
                    uppersolid_capstyle=None,
                    upperdash_joinstyle=None,
                    uppersolid_joinstyle=None,
                    
                    # lower hemisphere options & defaults
                    lowerlinewidth=1,
                    lowerlinestyle='-',
                    lowercolor=[0.6,0.6,0.6],
                    lowermarker=None,
                    lowermarkersize=None,
                    lowermarkeredgewidth=None,
                    lowermarkeredgecolor='none',
                    lowermarkerfacecolor='none',
                    lowermarkerfacecoloralt='none',
                    lowerfillstyle='full',
                    lowerdash_capstyle=None,
                    lowersolid_capstyle=None,
                    lowerdash_joinstyle=None,
                    lowersolid_joinstyle=None,
                    
                    # plot options & defaults
                    pickradius=5,
                    drawstyle=None,
                    markevery=None,
                    antialiased=None,
                    zorder=1,
                    **kwargs):

        plt.subplot(self._subplot)
        
        h = [ 1, 1,-1,-1, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0]
        k = [-1, 1,-1, 1, 0, 0, 0, 0, 1, 1,-1,-1, 0, 0, 1,-1, 0, 0]
        l = [ 0, 0, 0, 0,-1, 1,-1, 1,-1, 1,-1, 1, 0, 0, 0, 0, 1,-1]
        
        self.add_trace(miller(h,k,l), uc,
                    upperlinewidth=upperlinewidth,
                    upperlinestyle=upperlinestyle,
                    uppercolor=uppercolor,
                    uppermarker=uppermarker,
                    uppermarkersize=uppermarkersize,
                    uppermarkeredgewidth=uppermarkeredgewidth,
                    uppermarkeredgecolor=uppermarkeredgecolor,
                    uppermarkerfacecolor=uppermarkerfacecolor,
                    uppermarkerfacecoloralt=uppermarkerfacecoloralt,
                    upperfillstyle=upperfillstyle,
                    upperdash_capstyle=upperdash_capstyle,
                    uppersolid_capstyle=uppersolid_capstyle,
                    upperdash_joinstyle=upperdash_joinstyle,
                    uppersolid_joinstyle=uppersolid_joinstyle,
                    lowerlinewidth=lowerlinewidth,
                    lowerlinestyle=lowerlinestyle,
                    lowercolor=lowercolor,
                    lowermarker=lowermarker,
                    lowermarkersize=lowermarkersize,
                    lowermarkeredgewidth=lowermarkeredgewidth,
                    lowermarkeredgecolor=lowermarkeredgecolor,
                    lowermarkerfacecolor=lowermarkerfacecolor,
                    lowermarkerfacecoloralt=lowermarkerfacecoloralt,
                    lowerfillstyle=lowerfillstyle,
                    lowerdash_capstyle=lowerdash_capstyle,
                    lowersolid_capstyle=lowersolid_capstyle,
                    lowerdash_joinstyle=lowerdash_joinstyle,
                    lowersolid_joinstyle=lowersolid_joinstyle,
                    pickradius=pickradius,
                    drawstyle=drawstyle,
                    markevery=markevery,
                    antialiased=antialiased,
                    zorder=zorder,
                    **kwargs)

#------------------------------------------------------------------------------

def mplebsdmap(ovdat, prop):
    '''
    Plot an EBSD map using matplotlib.
    
    MPLEBSDMAP is slow. For fast, interactive plots, consider using ebsdmap.    
    
    BASIC PLOTTING COMMANDS:
    
    Example
    ------
    >>> import cryspy    
    >>> import matplotlib.pyplot as plt
    >>> data = loadang('my_file.ang')
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> col = mplebsdmap(data, data.pq)
    >>> ax.add_collection(col)
    >>> plt.axis('equal')
    >>> plt.show()
    '''    
    
    # convert the kernfaces and kernverts data to a sequence of xy tuples for
    # plotting a collection of polygons.
    tmpa = np.zeros([np.shape(ovdat.f)[1],np.shape(ovdat.f)[0],2])
    for i in range(0,np.shape(ovdat.f)[1]):
        tmpa[i,:,:] = ovdat.v[ovdat.f[:,i],:]
    tmpb = np.swapaxes(tmpa, 0, 1)
    verts = tuple(tuple(x) for x in tmpb)
    del tmpb, tmpa

    if np.amax(prop)>1:
        prop = prop/np.amax(prop)
        
    if np.size(np.shape(prop)) == 1:
        ncr = 3
    else:
        ncr = 1

    x = 0; del x # prevents redefine bug flag on next line    
    clr = [tuple(x) for x in np.tile(prop, [ncr,1]).T]
    
    t = tic()
    col = mplcollections.PolyCollection(verts, facecolors=clr, edgecolors=clr)
    toc(t)
    
    return col

#------------------------------------------------------------------------------

#def ebsdmap(ovdat, c, figure=None, axes=None):
#    '''
#    EBSDMAP uses the visvis library to interact with opengl for fast and
#    interactive plotting.
#    
#    For publication-quality plots, use of mplebsdmap is recommended.
#    
#    BASIC USAGE:
#    import ovlib as ov    
#    data = ov.loadang('my_file.ang')
#    ebsdmap(data,data.pq)  
#    
#    '''
#    h = figure
#    hax = axes
#    del figure, axes
#    
#    if h == None:
#        h = visvis.figure()
#    else:
#        visvis.figure(h)
#    
#    if hax == None:
#        hax = visvis.gca()
#    else:
#        visvis.axes(hax)
#    
#    if np.size(np.shape(c)) == 1:
#        c = np.tile(c,[3,1]).T
#    
#    if np.amax(c) > 1:
#        print 'c values must range between zero and one. \
#               Proceeding by normalizing values.'
#        c = c / np.amax(c)
#    
#    # To use visvis, we need to break our grid up into triangular 
#    # elements    
#    
#    if np.size(ovdat.cols) == 2: # hexagonal grid
#        c = np.reshape(np.tile(c.T,[12,1]).T, [-1, 3]) 
#    else: # square grid
#        c = np.reshape(np.tile(c.T,[6,1]).T, [-1, 3]) 
#
#    if hasattr(ovdat,'tv') and hasattr(ovdat,'vf'):        
#        tv = ovdat.tv
#        vf = ovdat.vf    
#    else:
#        tv, vf = triverts(ovdat.nr, ovdat.v, ovdat.f, ovdat.cols)
#    
#    m = visvis.mesh(vertices=tv, faces=vf)
#    visvis.processing.unwindFaces(m)
#    m.SetValues(c)
#    
#    # Flat painting
#    visvis.Mesh.shininess.fset(m,0)
#    visvis.Mesh.diffuse.fset(m,1)
#    visvis.Mesh.specular.fset(m,0)
#    
#    h.bgcolor = [1,1,1]    
#
#    hax.cameraType = '2d'
#    da = hax.daspect
#    hax.daspect = da[0], -abs(da[1]), da[2]
#    visvis.axis('off')
#    visvis.Axes.SetLimits(hax, rangeX=(np.amin(ovdat.x), np.amax(ovdat.x)), 
#                               rangeY=(np.amin(ovdat.y), np.amax(ovdat.y)))
#
#    
#    return h, hax

#------------------------------------------------------------------------------

#def ipfkey(symmetry, resolution=0.5, uc=None, 
#           trans='stereo', figure=None, axes=None):
#    ''' plot an ipf color key
#    
#    Plots an inverse pole figure color key for interpreting ebsd orientation
#    maps.
#
#    Parameters
#    -----------
#    pointgroup : point group identifier
#        the point group. default notation is numeric.
#    
#    Options
#    -------
#    resolution : float
#        the approximate angular resolution required
#        
#    uc : unit cell class object
#        a unit cell object containing the unit cell parameters
#        
#    trans : str
#        transformation: 'ea' for equal area; 'stereo' for stereographic
#        
#    notation : str
#        the notation of the point group input. Default is numeric.
#        
#    figure : visvis figure handle
#    
#    axes : visvis axis handle
#    
#    Returns
#    -------
#    h : visvis figure handle
#    
#    hax : visvis axis handle
#    '''
#
#    h = figure
#    hax = axes
#    del figure, axes
#    
#    if h == None:
#        h = visvis.figure()
#    else:
#        visvis.figure(h)
#    
#    if hax == None:
#        hax = visvis.gca()
#    else:
#        visvis.axes(hax)
#
#    if uc==None:
#        uc = unitcell()
#               
#    xproj, yproj, hem, edge = ipfgrid(symmetry.point_group_number, 
#                                      resolution=resolution, 
#                                      uc=None, trans=trans)
#    
#    cens, edg, tri, neigh = mpltriang.delaunay(xproj, yproj)
#    
#    zproj = np.zeros(np.shape(xproj))
#    tv2 = np.array([xproj, yproj, zproj]).T
#    vf2 = np.array(tri, dtype=np.uint32)
#    m = visvis.mesh(vertices=tv2, faces=vf2, verticesPerFace=3)
#    #visvis.processing.unwindFaces(m)
#
#    # This next part seems a little ridiculous... we are converting in and out of things perhaps unnecessarily
#    if trans=='ea':    
#        x,y,z = reveatransstandard( (xproj, yproj, hem) )
#    elif trans=='stereo':
#        x,y,z = revstereotransstandard( (xproj, yproj, hem) )
#                                
#    mil, dev = miller.from_cartesian( (x, y, z), uc, rationalizevals=False)
#    c = ipfcolor(mil, symmetry, uc)    
#    m.SetValues(c)
#    
#    # Flat painting
#    visvis.Mesh.shininess.fset(m, 0)
#    visvis.Mesh.diffuse.fset(m, 1)
#    visvis.Mesh.specular.fset(m, 0)
#    
#    h.bgcolor = [1,1,1]    
#    
#    hax.cameraType = '2d'
#    da = hax.daspect
#    hax.daspect = da[0], -abs(da[1]), da[2]
#    visvis.axis('off')
#    visvis.Axes.SetLimits(hax, rangeX=(np.amin(xproj), np.amax(xproj)), 
#                               rangeY=(np.amin(yproj), np.amax(yproj)))
#    
#    return h, hax

#------------------------------------------------------------------------------

def ipfcolor(m, symmetry, uc):
    """ color by directions in the external reference frame
    
    Parameters
    ----------
    m : miller object
    symmetry : symmetry object
    uc : unit cell object
    
    Returns
    -------
    c : n x 3 numpy array of floats between 0.0 and 1.0
        Arranged in columns of red, green, and blue components
    """
    # Put data in fundamental zone and convert to polar coordinates
    lcs = symmetry.laueclass()
    h = m.to_fundzone(lcs, uc)
    x, y, z = h.to_cartesian(uc)
    nrm = 1.0/vecarraynorm([x, y, z])
    x = x * nrm
    y = y * nrm
    z = z * nrm
    q, r, p = polar(x, y, z)
    #del r
    
    #visvis.figure()
    #x, y, z = cart(q, r, p)
    #visvis.plot(x,y,z,ms='.', ls='')
    #visvis.figure()
    
    # Treat m-3 as a special case
    if lcs.point_group_number == 28:
        pm    = p > np.pi/4.0
        p[pm] = np.pi/2.0 - p[pm]
    
    # get limits of the fundamental zone of the stereographic projection
    mintheta, maxtheta, minrho, maxrho = fundzonePF(lcs.point_group_number)
    
    if inspect.isfunction(maxrho):
        pmax = maxrho(q) # for m-3 and m-3m
    else:
        pmax = maxrho # for everything else
    
    # TSL colors the upper and lower halves identically for triclinic
    if lcs.point_group_number==1:
        q = np.arccos(np.cos(q))
        maxtheta = maxtheta * 0.5
    
    # normalize rho and theta
    p = p / (pmax + np.spacing(1))
    q = q / (maxtheta + np.spacing(1))
    
    scale = [2.6,2.0,2.0]#[2.6, 2.0, 2.0]
    pwr   = [0.8,1.0,1.0]#[0.8, 1.0, 1.0]
    shift = 0.4#0.4
     
    # assign red
    tmp = np.sqrt(scale[0])
    r = tmp * (tmp - np.absolute(q - shift)) * (1.0 - p)
    r[r > 1.0] = 1.0
    
    # assign green
    tmp = scale[1] * p
    tmp[tmp>1.0] = 1.0
    g = scale[1] * (1.0 - q) * tmp
    g[g > 1.0] = 1.0
    
    # assign blue
    tmp = scale[2] * p
    tmp[tmp>1.0] = 1.0
    b = scale[2] * q * tmp
    b[b > 1.0] = 1.0
    
    # adjust colors
    r = r**pwr[0]
    g = g**pwr[1]
    b = b**pwr[2]
    
    # fix out-of-range values
    r[r < 0] = 0.0
    g[g < 0] = 0.0
    b[b < 0] = 0.0
    
    return np.squeeze(np.array([r, g, b]).T)