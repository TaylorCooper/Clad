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

from numpy import *
import cv2, os, string, shutil, csv, time, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from pylab import * # plotting tools

from numpy import *


class mieCalc():
    """
    Description: Integrates amplitude functions
    Input: .MOV
    Output: Filtered and contact angle plots
    """
    
    def __init__(self):
            """ Initial parameters for mieCalc
            Input:
            Output:  
            """
            
            self.wavelength = 0.65 # Laser wavelength in um
            self.diameter = 1.0 # Particle diameter in um
            self.x = pi*self.diameter/self.wavelength
            self.complexIndex = 1.6+0.01j
            self.nang = 1000 # Max number of angles = 1000            
            
            # Geometry parameters
            self.photodiodeWindowThickness = 0.47 # Casing offset to chip in mm
            self.zChip = 1.7 # Casing offset to chip in mm
            
            # Z: Starting point in mm from edge of photodiode closest to laser
            # Scanning ranges Z: -10 to +10,  Y: 0.1 to +10
            self.scanZStartPt = -10
            # Length (in mm) traveled along laser axis from startPt
            self.scanZEndPt = 20 
            self.dZ = 0.5 # Recalculate integral every 0.5mm
            
            # Y: Starting point in mm from top of photodiode
            # Scanning ranges Y: 0.1 to +10
            self.scanYStartPt = 0.1
            # Length (in mm) traveled along laser axis from startPt
            self.scanYEndPt = 10 
            self.dY= 0.5 # Recalculate integral every dY
            
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
        
        # Assign extinction coefficients
        self.qext = qext
        self.qsca = qsca
        self.qback = qback
        self.gsca = gsca

    def scatterPlotter(self):
        """ Plot amplitude vs degrees 
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
        name = 'log[S1(blue) & S2(orange)]vs Angle, n- '
        name = name + str(self.complexIndex)
        name = name + ' x-' + str(self.x)[:4]
        ax.set_title(name,va='bottom', y=1.07, size=8)
        
        # Plot normalized intensity against angles
        ax = fig.add_subplot(212, polar=True)
        ax.set_theta_zero_location('W')
        ax.plot(self.angles, self.normIntensity, color='red')
        ax.yaxis.set_major_locator(tkr.MultipleLocator(0.0025))
        for yticka in ax.yaxis.get_major_ticks():
            yticka.label1.set_fontsize(6)
        ax.grid(True)
        name = 'Normalized Intensity vs Angle, n- '
        name = name + str(self.complexIndex)
        name = name + ' x-' + str(self.x)[:4]
        ax.set_title(name,va='bottom', y=1.07, size=8)
        
        plt.subplots_adjust(hspace=0.3)
        savefig('mieScatterPlot.png', dpi=350, bb_inches='loose')
        clf()
        
    def scanXY(self):
        """ Scan over YX ranges identified in init
            Input: self.bhmie()
        """

        # Get arrays for z and y
        arrZ = arange(self.scanZStartPt,self.scanZEndPt,self.dZ)
        arrY = arange(self.scanYStartPt,self.scanYEndPt,self.dY)
        
        for y in arrY:
            
            print '--------------',y,'--------------'
           
            for z in arrZ:
                
                if z <= 0:
                    print 'z<0',
                    theta1 = pi/2 + arctan(abs(z)/y)
                    theta2 = pi/2 + arctan((abs(z)+self.zChip)/y)
                elif z > 0 and z <= self.zChip:
                    print 'z>0',
                    theta1 = arctan(y/abs(z))
                    theta2 = pi/2 + arctan((abs(z)+self.zChip)/y)
                elif z > self.zChip:
                    print 'z<<0',
                    theta1 = arctan(y/abs(z))
                    theta2 = arctan(y/(abs(z)-self.zChip))
                else:
                    print 'Impossibru!'
                    
                print z, theta1*180/pi, theta2*180/pi     
                
        

mc = mieCalc()
mc.bhmie()
#mc.scatterPlotter()

mc.scanXY()

        

