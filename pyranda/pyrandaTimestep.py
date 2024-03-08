################################################################################
# Copyright (c) 2018, Lawrence Livemore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# LLNL-CODE-749864
# This file is part of pyranda
# For details about use and distribution, please read: pyranda/LICENSE
#
# Written by: Britton J. Olson, olson45@llnl.gov
################################################################################
import numpy 
from .pyrandaPackage import pyrandaPackage


class pyrandaTimestep(pyrandaPackage):
    """
    Case physics package module for adding new physics packages to pyranda
    """
    def __init__(self,pysim):

        PackageName = 'Timestep'
        pyrandaPackage.__init__(self,PackageName,pysim)

        self.dx = pysim.mesh.d1
        self.dy = pysim.mesh.d2
        self.dz = pysim.mesh.d3
        self.pysim = pysim
        self.GridLen = pysim.mesh.GridLen
        self.IB_offset = pysim.IB_offset
        
    def get_sMap(self):
        """
        String mappings for this package.  Packages added to the main
        pyranda object will check this map
        """
        sMap = {}
        sMap['dt.IBsqroot('] = "self.packages['Timestep'].IBsqroot("
        sMap['dt.courant('] = "self.packages['Timestep'].courant("
        sMap['dt.diff('] = "self.packages['Timestep'].diff("
        sMap['dt.diffDir('] = "self.packages['Timestep'].diffDir("
        self.sMap = sMap

    def IBsqroot(self, val):
        val[:,:self.IB_offset,:] *= 0.
        if numpy.min(val) < 0.:
            import pdb; pdb.set_trace()
        return numpy.sqrt(val)

    def courant(self,u,v,w,c):

        # Compute the dt for the courant limit
        if self.pysim.mesh.coordsys == 3:
            dAdx = self.pysim.getVar("dAx")
            dAdy = self.pysim.getVar("dAy")
            dBdx = self.pysim.getVar("dBx")
            dBdy = self.pysim.getVar("dBy")
            magA = numpy.sqrt( dAdx*dAdx + dAdy*dAdy )
            magB = numpy.sqrt( dBdx*dBdx + dBdy*dBdy )
            uA = ( u*dAdx + v*dAdy ) / magA
            uB = ( u*dBdx + v*dBdy ) / magB
            vrate = ( numpy.abs(uA) / self.pysim.getVar('d1') +
                      numpy.abs(uB) / self.pysim.getVar('d2') )

        else:
            vrate = ( numpy.abs(u) / self.dx +
                      numpy.abs(v) / self.dy +
                      numpy.abs(w) / self.dz )
        
        vrate[:,:self.IB_offset,:] *= 0.
        
        crate = numpy.abs(c) / self.GridLen

        dt_max = 1.0 / self.pyranda.PyMPI.max3D(vrate + crate)

        return dt_max
        
        

    def diff(self,bulk,density):

        delta = self.GridLen
        drate = density * delta * delta / numpy.maximum( 1.0e-12, bulk )

        drate[:,:self.IB_offset,:] += 1.

        dt_max = self.pyranda.PyMPI.min3D( drate )

        return dt_max


    def diffDir(self,bulk,density,delta):

        drate = density * delta * delta / numpy.maximum( 1.0e-12, bulk )

        drate[:,:self.IB_offset,:] += 1.

        dt_max = self.pyranda.PyMPI.min3D( drate )

        return dt_max

