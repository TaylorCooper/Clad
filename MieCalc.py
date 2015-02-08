# -*- coding: utf-8 -*-

"""
    Author:
        Taylor Cooper
    Description:
        Integrates amplitude functions S1 and S2 over angles prescribed           
    Date Created:
        2015.02.03
    
    Arguments and Inputs:
        R - Particle radius
        lmda - Laser wavelength
        Y - Heigh for Xsweep
        xStart - Distance from 
        xLen - 
        
        ### Xsweep / Ysweep would be useful
        
                        
    Outputs:
        .txt containing frameNumber, leftCA, rightCA, diameter
    
    
    Dependencies:
        numpy, cv2, os, string, pylab, shutil, csv, time
    
    Limitations:
        
    
    Pending Major Changes:

    History:                  
    --------------------------------------------------------------
    Date:    
    Author:    
    Modification:    
     --------------------------------------------------------------
"""

from numpy import * # Ugly but done to save time / avoid reworking bhmie
import os, string, shutil, csv, time, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from pylab import * # plotting tools

# Require for 3D plot
from mpl_toolkits.mplot3d import axes3d, Axes3D

DEBUG = False

class mieCalc():
    """
    Description: Integrates amplitude functions
    Input: wavelength in um, diamter in um, complex index of refrac.
    Methods:
        bhmie - Calculate amplitude functions and extinction coefficients
        ampFuncPlotter - Plots amplitude functions
        xyIntensityPlotter - Plots surface indicating 
    """
    
    def __init__(self, path, wavelength=0.65, diameter=1.0, index=1.6+0.1j):
            """ Init method                 
            """
            
            self.path = path # Working directory
            
            self.wavelength = wavelength # Laser wavelength in um
            self.diameter = diameter # Particle diameter in um
            self.x = pi*self.diameter/self.wavelength
            self.complexIndex = index
            self.nang = 1000 # Max number of angles = 1000            
            
            # Geometry parameters
            self.photodiodeWindowThickness = 0.47 # Casing offset to chip in mm
            self.zChip = 2.7 # Casing offset to chip in mm
            
            # Z: Starting point in mm from edge of photodiode closest to laser
            # Scanning ranges Z: -10 to +10,  Y: 0.1 to +10
            self.scanZStartPt = -10
            # Length (in mm) traveled along laser axis from startPt
            self.scanZEndPt = 10 + self.zChip
            self.dZ = 0.2 # Recalculate integral every 0.5mm
            
            # Y: Starting point in mm from top of photodiode
            # Scanning ranges Y: 0.1 to +10
            self.scanYStartPt = 0.1
            # Length (in mm) traveled along laser axis from startPt
            self.scanYEndPt = 8
            self.dY= self.dZ # Recalculate integral every dY
            
            # Mie Parameters
            self.s1 = None
            self.s2 = None
            self.normIntensity = None           
            self.angles = None
            self.qext = None
            self.qsca = None
            self.qback = None
            self.gsca = None
            
            


    def bhmie(self):
        """ This file from mie.m, see http://atol.ucsd.edu/scatlib/index.htm
        Bohren and Huffman originally published the code in their book on 
        light scattering
        
        Calculation based on Mie scattering theory  
        input:
            x - size parameter = k*radius = 2pi/lambda * radius   
                (lambda is the wavelength in the medium around the scatterers)
            refrel - refraction index (n in complex form) i.e.:  1.5+0.02*j;
            nang   - number of angles for S1 and S2 function from 0 to pi
        output:
            S1, S2 - function which correspond to the (complex) phase functions
            angles - Angles associated with S1,S2
            Qext   - extinction efficiency
            Qsca   - scattering efficiency 
            Qback  - backscatter efficiency
            gsca   - asymmetry parameter
        """
    
        # Assign class variables
        x = self.x
        refrel = self.complexIndex
        nang = self.nang
    
        nmxx=150000
        
        s1_1=zeros(nang,dtype=complex128)
        s1_2=zeros(nang,dtype=complex128)
        s2_1=zeros(nang,dtype=complex128)
        s2_2=zeros(nang,dtype=complex128)
        pi=zeros(nang)
        tau=zeros(nang)
        
        if (nang > 1000):
            print ('error: nang > mxnang=1000 in bhmie')
            return
        
        # Require NANG>1 in order to calculate scattering intensities
        if (nang < 2):
            print ('error: nang < mnnang=2 in bhmie')
            return
        
        pii = 4.*arctan(1.)
        dx = x
          
        drefrl = refrel
        y = x*drefrl
        ymod = abs(y)
        
        
        #    Series expansion terminated after NSTOP terms
        #    Logarithmic derivatives calculated from NMX on down
        
        xstop = x + 4.*x**0.3333 + 2.0
        nmx = max(xstop,ymod) + 15.0
        nmx=fix(nmx)
         
        # BTD experiment 91/1/15: add one more term to series and compare resu<s
        #      NMX=AMAX1(XSTOP,YMOD)+16
        # test: compute 7001 wavelen>hs between .0001 and 1000 micron
        # for a=1.0micron SiC grain.  When NMX increased by 1, only a single
        # computed number changed (out of 4*7001) and it only changed by 1/8387
        # conclusion: we are indeed retaining enough terms in series!
        
        nstop = int(xstop)
        
        if (nmx > nmxx):
            print ( "error: nmx > nmxx=%f for |m|x=%f" % ( nmxx, ymod) )
            return
        
        dang = pii/ (nang-1) # 180 degrees / number of angles
        
    
        amu=arange(0.0,nang,1)
        amu=cos(amu*dang)
    
        pi0=zeros(nang)
        pi1=ones(nang)
        
        # Logarithmic derivative D(J) calculated by downward recurrence
        # beginning with initial value (0.,0.) at J=NMX
        
        nn = int(nmx)-1
        d=zeros(nn+1)
        for n in range(0,nn):
            en = nmx - n
            d[nn-n-1] = (en/y) - (1./ (d[nn-n]+en/y))
        
        #*** Riccati-Bessel functions with real argument X
        #    calculated by upward recurrence
        
        psi0 = cos(dx)
        psi1 = sin(dx)
        chi0 = -sin(dx)
        chi1 = cos(dx)
        xi1 = psi1-chi1*1j
        qsca = 0.
        gsca = 0.
        p = -1
        
        for n in range(0,nstop):
            en = n+1.0
            fn = (2.*en+1.)/(en* (en+1.))
        
        # for given N, PSI  = psi_n        CHI  = chi_n
        #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
        #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
        # Calculate psi_n and chi_n
            psi = (2.*en-1.)*psi1/dx - psi0
            chi = (2.*en-1.)*chi1/dx - chi0
            xi = psi-chi*1j
        
        #*** Store previous values of AN and BN for use
        #    in computation of g=<cos(theta)>
            if (n > 0):
                an1 = an
                bn1 = bn
        
        #*** Compute AN and BN:
            an = (d[n]/drefrl+en/dx)*psi - psi1
            an = an/ ((d[n]/drefrl+en/dx)*xi-xi1)
            bn = (drefrl*d[n]+en/dx)*psi - psi1
            bn = bn/ ((drefrl*d[n]+en/dx)*xi-xi1)
    
        #*** Augment sums for Qsca and g=<cos(theta)>
            qsca += (2.*en+1.)* (abs(an)**2+abs(bn)**2)
            gsca += ((2.*en+1.)/ (en* (en+1.)))*( real(an)* real(bn)+imag(an)*imag(bn))
        
            if (n > 0):
                gsca += ((en-1.)* (en+1.)/en)*( real(an1)* real(an)+imag(an1)*imag(an)+real(bn1)* real(bn)+imag(bn1)*imag(bn))
        
        
        #*** Now calculate scattering intensity pattern
        #    First do angles from 0 to 90
            pi=0+pi1    # 0+pi1 because we want a hard copy of the values
            tau=en*amu*pi-(en+1.)*pi0
            s1_1 += fn* (an*pi+bn*tau)
            s2_1 += fn* (an*tau+bn*pi)
        
        #*** Now do angles greater than 90 using PI and TAU from
        #    angles less than 90.
        #    P=1 for N=1,3,...% P=-1 for N=2,4,...
        #   remember that we have to reverse the order of the elements
        #   of the second part of s1 and s2 after the calculation
            p = -p
            s1_2+= fn*p* (an*pi-bn*tau)
            s2_2+= fn*p* (bn*pi-an*tau)
    
            psi0 = psi1
            psi1 = psi
            chi0 = chi1
            chi1 = chi
            xi1 = psi1-chi1*1j
        
        #*** Compute pi_n for next value of n
        #    For each angle J, compute pi_n+1
        #    from PI = pi_n , PI0 = pi_n-1
            pi1 = ((2.*en+1.)*amu*pi- (en+1.)*pi0)/ en
            pi0 = 0+pi   # 0+pi because we want a hard copy of the values
        
        #*** Have summed sufficient terms.
        #    Now compute QSCA,QEXT,QBACK,and GSCA
    
        #   we have to reverse the order of the elements of the second part of s1 and s2
        s1=concatenate((s1_1,s1_2[-2::-1]))
        s2=concatenate((s2_1,s2_2[-2::-1]))
        gsca = 2.*gsca/qsca
        qsca = (2./ (dx*dx))*qsca
        qext = (4./ (dx*dx))* real(s1[0])
    
        # more common definition of the backscattering efficiency,
        # so that the backscattering cross section really
        # has dimension of length squared
        qback = 4*(abs(s1[2*nang-2])/dx)**2    
        #qback = ((abs(s1[2*nang-2])/dx)**2 )/pii  #old form
    
        # Create angles in radians to match the 1999pts of s1/s2, the negative
        # concatenation is required because of the way S1,S2 are shaped
        self.angles = arange(0.0,nang,1)*dang # Angles in degrees for s1, s2        
        self.angles = np.concatenate((self.angles,self.angles*-1.0),axis=0)[:-1]
        self.s1 = s1
        self.s2 = s2
        
        # Create an array with normalized intensity from s1 & s2, this array is 
        # the total intensity reflected from natural light normalized to 1 
        intensity = ((abs(s1)**2)/2 + (abs(s2)**2)/2)/(x**2)
        self.normIntensity = intensity/np.sum(intensity)
        
        # Assign extinction coefficients, currently these aren't used
        self.qext = qext
        self.qsca = qsca
        self.qback = qback
        self.gsca = gsca
        

    def ampFuncPlotter(self):
        """ Input: bhmie
            Output: polar plot amplitude vs degrees 
        """

        # Plot parallel and perpendicular amplitude functions against angles
        fig = plt.figure(figsize=(4,6))
        ax = fig.add_subplot(211, polar=True)
        ax.set_theta_zero_location('W') # Rotate polar axis to match my diagrams
        ax.plot(self.angles, log(self.s1), color='blue') # Para
        ax.plot(self.angles, log(self.s2), color='orange') # Perp
        ax.yaxis.set_major_locator(tkr.MultipleLocator(2))
        for yticka in ax.yaxis.get_major_ticks():
            yticka.label1.set_fontsize(6)
        ax.grid(True)
        name = 'log[S1(b) & S2(o)]  d='
        name = name + str(self.diameter)[:4]
        name = name + 'um  n=' + str(self.complexIndex)
        ax.set_title(name,va='bottom', y=1.07, size=7)
        
        # Plot normalized intensity against angles
        ax = fig.add_subplot(212, polar=True)
        ax.set_theta_zero_location('W')
        ax.plot(self.angles, self.normIntensity, color='red')
        ax.yaxis.set_major_locator(tkr.MultipleLocator(0.0025))
        for yticka in ax.yaxis.get_major_ticks():
            yticka.label1.set_fontsize(6)
        ax.grid(True)
        
        # Set title
        name = 'Normalized Intensity  d='
        name = name + str(self.diameter)[:4]
        name = name + 'um  n=' + str(self.complexIndex)
        ax.set_title(name,va='bottom', y=1.07, size=7)
        
        # Save figure
        plt.subplots_adjust(hspace=0.3)
        path = self.path + 'd=' + str(self.diameter)[:4]
        path = path + 'um n=' + str(self.complexIndex) 
        path = path + ' mieScatterPlot' + '.png'
        
        print path
        savefig(path, dpi=350, bb_inches='loose')
        clf()
        
       
    def integrator(self, theta1, theta2):
        """ Takes 2 angles and sums the intensity between them
        """
        
        if theta1 > theta2:
            print 'Error theta1 > theta2'
            return None
        
        sum = 0
        
        for i,ang in enumerate(self.angles):
            if ang < theta1:
                continue
            elif ang >= theta1 and ang <= theta2:
                sum = sum + self.normIntensity[i]
            elif ang > theta2:
                return sum
            
        print 'Error thetaMax < theta2'
        return None
                
     
    def xyIntensityPlotter(self):
        """ Scan over YX ranges identified in init
            Input: self.bhmie()
        """
        

        # Get arrays for z and y
        arrZ = arange(self.scanZStartPt,self.scanZEndPt,self.dZ)
        arrY = arange(self.scanYStartPt,self.scanYEndPt,self.dY)

        # Create arrays for 3D plot
        zAxis = []
        
        for y in arrY:
                        
            if DEBUG: print '--------------',y,'--------------'
            cols = []
           
            for z in arrZ:
                
                if z <= 0:
                    if DEBUG: print 'z<0',
                    theta1 = pi/2 + arctan(abs(z)/y)
                    theta2 = pi/2 + arctan((abs(z)+self.zChip)/y)
                elif z > 0 and z <= self.zChip:
                    if DEBUG: print 'z>0',
                    theta1 = arctan(y/abs(z))
                    theta2 = pi/2 + arctan((abs(z)+self.zChip)/y)
                elif z > self.zChip:
                    if DEBUG: print 'z<<0',
                    theta1 = arctan(y/abs(z))
                    theta2 = arctan(y/(abs(z)-self.zChip))
                else:
                    if DEBUG: print 'Impossibru!'
                    
                # Intensity radiated onto sensing area of PD
                observedIntensity = self.integrator(theta1, theta2)
                cols.append(observedIntensity*100) # Convert to percentage

                if DEBUG: print z, theta1*180/pi, theta2*180/pi
                
            zAxis.append(cols)
                
        # 3D surface plot of values
        fig = plt.figure()
        ax = Axes3D(fig) # Ugly crap required for 3D plot
        xAxis, yAxis = meshgrid(arrZ,arrY) # Need 2D arrays
        surf = ax.plot_surface(xAxis,yAxis,zAxis, rstride=1, cstride=1, 
                        cmap=cm.coolwarm, linewidth=0)
        ax.view_init(elev=7, azim=120) # Set view point
        
        # Add a side bar to indicate color intensity and title
        fig.colorbar(surf, shrink=0.5, aspect=5)
        name = 'Light Incident on PD vs Particle Location d='
        name = name + str(self.diameter)[:4]
        name = name + 'um  n=' + str(self.complexIndex)
        name = name + '\n\n Surface Grid Size=' + str(self.dZ)[:3] +'mm'
        ax.set_title(name,va='bottom', y=0.97, size=12)
        ax.set_xlabel('Distance From PD Edge (mm)')
        ax.set_ylabel('Height Above PD (mm)')
        ax.set_zlabel('Light Intensity Incident on PD (%)')
        
        # Save figure
        path = self.path + 'd=' + str(self.diameter)[:4]
        path = path + 'um n=' + str(self.complexIndex) 
        path = path + ' xyIntensityPlot' + '.png'
        print path
        savefig(path, dpi=350)
        clf()
                

diameterRange = arange(0.5,10,0.5)
diameterRange = insert(diameterRange,0,0.25)

# Indices by material in order from first to last
# 1.49+0.0j - Dioctyl phthalate (DOP) at unknown wavelength
# 1.60+0.0j - Polystyrene latex (PSL) at 589nm
# 1.54+0.5j - Coal dust at unknown wavelength
indexRange = [1.49+0.0j, 1.60+0.0j, 1.54+0.5j]
path = "D:\\GitHub\\workspace\\Clad\\output\\"


for n in indexRange:
    for d in diameterRange:

        # Generate plots for all possible configuratins of values
        mc = mieCalc(path,diameter=d,index=n)
        mc.bhmie()
        mc.ampFuncPlotter()
        mc.xyIntensityPlotter()

        

