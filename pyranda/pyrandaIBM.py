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
import scipy.optimize
from mpi4py import MPI
from .pyrandaPackage import pyrandaPackage

immersed_iter = 2
immersed_CFL = 0.5
immersed_EPS = 0.1

class pyrandaIBM(pyrandaPackage):

    def __init__(self,pysim):

        PackageName = 'IBM'
        pyrandaPackage.__init__(self,PackageName,pysim)

    def get_sMap(self):
        sMap = {}
        sMap['ibmWall('] = "self.packages['IBM'].ibmWall("
        sMap['ibmWM('] = "self.packages['IBM'].ibmWallModel("
        sMap['ibmV('] = "self.packages['IBM'].ibmVel("
        sMap['ibmS('] = "self.packages['IBM'].ibmS("
        self.sMap = sMap

    def ibmWallModel(self,vel,phi,gphi,BL_params,local=False,phivar=None):
        nu, hwm, deltax = BL_params

        gridlen = self.pyranda.mesh.GridLen
        #dx = self.pyranda.dx
        #dy = self.pyranda.dy
        #dz = self.pyranda.dz
        #if phi.shape[-1] == 1:
        #    dz = dx
        lens = gridlen * immersed_EPS
        lo = self.pyranda.PyMPI.chunk_3d_lo
        ax, ay, az = phi.shape

        [hind,hrem,hdy] = self.pyranda.userDefined['inverseMesh'](hwm)

        u = vel[0]
        v = vel[1]
        w = vel[2]

        u0 = u.copy()*0.
        Uwm = u.copy()*0.
        utau = u.copy()*0.

        phix = gphi[0]
        phiy = gphi[1]
        phiz = gphi[2]

        mask = abs(phi - hwm) <= hdy
        u_h = u[mask]
        v_h = v[mask]
        w_h = w[mask]
        if len(u_h) == 0:
            u_0 = u_h.copy()
            U_wm = u_h.copy()
            u_tau = u_h.copy()
            u1h = u_h.copy()
            u2h = u_h.copy()
            u3h = u_h.copy()

        else:

            phix_h = phix[mask]
            phiy_h = phiy[mask]
            phiz_h = phiz[mask]
      
            # remove normal velocity and get resulting velocity profile
            norm = u_h*phix_h + v_h*phiy_h + w_h*phiz_h
            u_h -= norm * phix_h
            v_h -= norm * phiy_h
            w_h -= norm * phiz_h
      
            # compute parallel flow velocity profile at phi~=hwm
            U_wm = numpy.sqrt(u_h*u_h + v_h*v_h + w_h*w_h)

            # need to save off the tangential velocity directional vector at this phi value
            tol = 1e-10
            u1h = u_h / (U_wm + tol)
            u2h = v_h / (U_wm + tol)
            u3h = w_h / (U_wm + tol)

            # solve for u_tau
            kappa = 0.41
            A = 5.0
      
            # root finding for whole array at once isn't working for some reason
            u_tau = numpy.zeros_like(u_h)
            for j in range(0,len(U_wm),40):
                u_wm = U_wm[j:j+40]
                guess = 1e-14*numpy.ones_like(u_wm)
                def g(ut):
                    val = u_wm/ut - 1/kappa*numpy.log(hwm*ut/nu)-A
                    valprime = -1/ut * (u_wm/ut - 1/kappa)
                    return val, numpy.diag(valprime)
                sol = scipy.optimize.root(g,guess,jac=True)
                if not sol.success:
                    for k in range(40):
                        if j+k == len(U_wm):
                            break
                        u_wm = U_wm[j+k]
                        guessk = guess[k]
                        single_sol = scipy.optimize.root(g,guessk,jac=True)
                        if not single_sol.success:
                            print(u_wm,guessk,j,k)
                            #import pdb
                            #pdb.set_trace()
                            
                        u_tau[j+k] = single_sol.x[0]
                else:
                    u_tau[j:j+40] = sol.x
                
            # solve for u*_wm(0)
            u_0 = u_tau/kappa * (numpy.log(u_tau*deltax/nu)-1) + u_tau*A
            #import pdb; pdb.set_trace()
            #import pdb;pdb.set_trace()
            #import matplotlib.pyplot as plt
            #f = lambda y: u_tau[0]/kappa*numpy.log(u_tau[0]*y/nu)+u_tau[0]*A
            #plt.plot([0.,deltax],[u_0[0],f(deltax)])
            #dom = numpy.linspace(0.01,10.,1000)
            #plt.plot(dom, f(dom))
            #plt.plot([0.,deltax, hwm],[u_0[0],f(deltax),f(hwm)],'*')
            #plt.show()
        if False:
            u0[mask] = u_0
            tvec = []
            for ind,vals in enumerate([u1h,u2h,u3h]):
                tangent_vec = numpy.zeros_like(u0)
                tangent_vec[mask] = vals
                tangent_vec = self.modified_smoother(phi-hwm, phix, phiy, phiz, tangent_vec)
                tvec.append(tangent_vec)
                
            u0 = self.modified_smoother(phi-hwm, phix, phiy, phiz, u0)
            
        else: 
    
            [hind,hrem,hdy] = self.pyranda.userDefined['inverseMesh'](hwm)
            
            # now put this into a global array
            nx, ny, nz = self.pyranda.mesh.options['nn']
            local_u0 = numpy.zeros((nx, ny, nz))
            lo = self.pyranda.PyMPI.chunk_3d_lo
            hi = self.pyranda.PyMPI.chunk_3d_hi
            local_u0[lo[0]:hi[0]+1,lo[1]:hi[1]+1,lo[2]:hi[2]+1][mask] = u_0

            tangent_vec = numpy.zeros((3,nx,ny,nz))
            for ind,vals in enumerate([u1h,u2h,u3h]):
                tangent_vec[ind,lo[0]:hi[0]+1,lo[1]:hi[1]+1,lo[2]:hi[2]+1][mask] = vals

            utau[mask] = u_tau
            Uwm[mask] = U_wm

            #tmp0 = (hwm-phi)*phix
            #tmp1 = (hwm-phi)*phiy
            #tmp2 = (hwm-phi)*phiz

            #shift0 = (tmp0/dx).astype(int)
            #shift1 = (tmp1/dy).astype(int)
            #shift2 = (tmp2/dz).astype(int)

            #offset0 = (tmp0 - dx*shift0)/dx
            #offset1 = (tmp1 - dy*shift1)/dy
            #offset2 = (tmp2 - dz*shift2)/dz

            offset = hrem / hdy

            xdisp, zdisp = 0., 0.
            ydisp = offset

            def interp(C):#,i,j,k):
                #xdisp = offset0[i,j,k]
                #ydisp = offset1[i,j,k]
                #zdisp = offset2[i,j,k]
                if min(C.shape) > 1:
                    c00 = C[0,0,0]*(1-xdisp)+C[1,0,0]*xdisp
                    c01 = C[0,0,1]*(1-xdisp)+C[1,0,1]*xdisp
                    c10 = C[0,1,0]*(1-xdisp)+C[1,1,0]*xdisp
                    c11 = C[0,1,1]*(1-xdisp)+C[1,1,1]*xdisp

                    c0 = c00*(1-ydisp)+c10*ydisp
                    c1 = c01*(1-ydisp)+c11*ydisp

                    c = c0*(1-zdisp)+c1*zdisp
                else:
                    if sum(C.shape) == 3:
                        c = C[0,0,0]
                    elif sum(C.shape[:-1]) == 2:
                        c = C[0,0,0]*(1-zdisp)+C[0,0,1]*zdisp
                    elif sum(C.shape[1:]) == 2:
                        c = C[0,0,0]*(1-xdisp)+C[1,0,0]*xdisp
                    elif sum(C.shape) == 4:
                        c = C[0,0,0]*(1-ydisp)+C[0,1,0]*ydisp
                    elif C.shape[0] == 1:
                        c0 = C[0,0,0]*(1-zdisp)+C[0,0,1]*zdisp
                        c1 = C[0,1,0]*(1-zdisp)+C[0,1,1]*zdisp

                        c = c0*(1-ydisp)+c1*ydisp
                    elif C.shape[1] == 1:
                        c0 = C[0,0,0]*(1-xdisp)+C[1,0,0]*xdisp
                        c1 = C[0,0,1]*(1-xdisp)+C[1,0,1]*xdisp

                        c = c0*(1-zdisp)+c1*zdisp
                    else:
                        c0 = C[0,0,0]*(1-xdisp)+C[1,0,0]*xdisp
                        c1 = C[0,1,0]*(1-xdisp)+C[1,1,0]*xdisp

                        c = c0*(1-ydisp)+c1*ydisp
                return c

            global_u0 = self.pyranda.PyMPI.comm.allreduce(local_u0,op=MPI.SUM)
            global_t_vec = self.pyranda.PyMPI.comm.allreduce(tangent_vec,op=MPI.SUM)
            tvec = numpy.zeros((3,ax,ay,az))
            # would be more efficient to mask
            for i in range(max(ax,1)):
                for k in range(max(az,1)):
                    ii = lo[0]+i
                    kk = lo[2]+k
                    end = 2 - (numpy.array([ii,hind,kk],dtype=int)==numpy.array(u0.shape)-1)
                    #end = 2 - (numpy.array([ii,jj,kk],dtype=int)==numpy.array(u0.shape)-1)
                    try:
                        neighbs = global_u0[ii:ii+end[0],hind:hind+end[1],kk:kk+end[2]]
                        #neighbs = global_u0[ii:ii+end[0],jj:jj+end[1],kk:kk+end[2]]
                    except:
                        print(ii,hind,kk,end,type(global_u0),end[0],end[1],end[2])
                        #print(ii,jj,kk,end,type(global_u0),end[0],end[1],end[2])
                    val = interp(neighbs)
                    tvec_vals = []
                    for d in range(3):
                        tvec_neighbs = global_t_vec[d,ii:ii+end[0],hind:hind+end[1],kk:kk+end[2]]
                        tvec_vals.append(interp(tvec_neighbs))
                    #for j in range(max(ay,1)):
                    for j in range(max(0,self.pyranda.IB_offset-5),hind):
                        #if phi[i,j,k] < lens[i,j,k] and :
                        u0[i,j,k] = val
                        for d in range(3):
                            tvec[d,i,j,k] = tvec_vals[d]
        [v1, v2, v3] = self.wm_velocity( phi ,phix,phiy,phiz,
                                   u,v,w,lens,u0,tvec,new=False,phivar=phivar)
        #if numpy.linalg.norm(self.pyranda.PyMPI.chunk_3d_lo-numpy.array([0,20,0]))==0.:
        #    import pdb;pdb.set_trace()
        return [v1, v2, v3, utau, Uwm, u0]

    def modified_smoother(self, SDF, gDx, gDy, gDz, val_in, epsi=0.0, new=False):
        val = val_in * 1.0
        GridLen = self.pyranda.mesh.GridLen
        alpha = 0.5  # < .4 is unstable!
        for i in range(immersed_iter):
            [tvx,tvy,tvz] = self.pyranda.grad(val)
            term = tvx*gDx+tvy*gDy+tvz*gDz
            val = numpy.where( SDF <= epsi , val + immersed_CFL*GridLen*term , val )
            Tval = self.pyranda.gfilter(val)
            Tval = Tval * alpha + val * (1.0 - alpha)
            val = numpy.where( SDF <= epsi , Tval, val )        
        return val
        
    def wm_velocity(self,SDF,gDx,gDy,gDz,v1_in,v2_in,v3_in,lens,u0,tvec,new=False,phivar=None):
        #if numpy.max(self.pyranda.PyMPI.chunk_3d_lo)==0:
        #    x = self.pyranda.var('meshx').data[:,:,0]
        #    y = self.pyranda.var('meshy').data[:,:,0]
        #    import matplotlib.pyplot as plt
        #    def plot(v):
        #        for j in range(u0.shape[0]):
        #            vv = self.pyranda.var(v).data[j,:,0]
        #            plt.plot(y[j,:],vv)
        #        plt.show()
        #        
        #    def vis(v):
        #        vv = self.pyranda.var(v).data[:,:,0]
        #        plt.contourf(x,y,vv,cmap='jet')
        #        plt.colorbar()
        #        plt.show()

        #    def look2d(v):
        #        vv = v[:,:,0]
        #        plt.contourf(x,y,vv,cmap='jet')
        #        plt.colorbar()
        #        plt.show()
        #    def look(vlist):
        #        colors = ['k','b','r','C3','C4']
        #        for c,v in zip(colors,vlist):
        #            for j in range(v.shape[0]):
        #                vv = v[j,:,0]
        #                plt.plot(y[j,:],vv,c=c)
        #        plt.show()
        #    import pdb; pdb.set_trace()

        if self.pyranda.cycle == 0:
            u0_smooth = self.smooth_terrain(SDF,gDx,gDy,gDz,u0,lens,new=False)
            for k in range(50): 
                u0_smooth = self.smooth_terrain(SDF,gDx,gDy,gDz,u0_smooth,lens,new=False)
        else:
            u0[SDF < lens] = self.u0_smooth[SDF < lens]
            u0_smooth = self.smooth_terrain(SDF,gDx,gDy,gDz,u0,lens,new=False)
        self.u0_smooth = u0_smooth

        # implement slip velocity and enforce wall model bc
        # essentially copying the slip velocity function with slip=True
        v1 = v1_in*1.0
        v2 = v2_in*1.0
        v3 = v3_in*1.0

        if phivar:
            v1_phi = phivar[0]
            v2_phi = phivar[1]
            v3_phi = phivar[2]

            # Transform to interface velocity
            v1 -= v1_phi
            v2 -= v2_phi
            v3 -= v3_phi

        v1 = self.smooth_terrain(SDF,gDx,gDy,gDz,v1,0.0,new=new)
        v2 = self.smooth_terrain(SDF,gDx,gDy,gDz,v2,0.0,new=new)
        v3 = self.smooth_terrain(SDF,gDx,gDy,gDz,v3,0.0,new=new)

        normal = v1*gDx+v2*gDy+v3*gDz

        vn =  numpy.where( SDF < lens, normal, 0.0 )
            
        # Remove normal velocity
        v1 = v1 - vn*gDx
        v2 = v2 - vn*gDy
        v3 = v3 - vn*gDz

        # create tangent vector
        tangent = numpy.sqrt(v1*v1+v2*v2+v3*v3)

        tol = 1e-10
        u1 = v1 / (tangent+tol)
        u2 = v2 / (tangent+tol)
        u3 = v3 / (tangent+tol)

        # set velocity to zero for phi < 0
        interior_mask = numpy.where( SDF < lens, 0.0, 1.0)
        v1 = v1*interior_mask
        v2 = v2*interior_mask
        v3 = v3*interior_mask

        # Compute linear velocity through zero level
        tmp = numpy.where( SDF < lens, 0.0 , normal/(SDF+tol) )
        tmp_n = self.smooth_terrain(SDF,gDx,gDy,gDz,tmp,lens,new=False)
        vn = numpy.where( SDF < lens, tmp_n*SDF, 0.0 )

        # Compute the linear velocity through the wall model bc
        tmp = numpy.where( SDF < lens, u0,0.0)# , tangent/(SDF+tol))
        tmp_t0 = self.smooth_terrain(-SDF,-gDx,-gDy,-gDz,tmp,lens,new=False)
        tmp_t1 = self.smooth_terrain(SDF,gDx,gDy,gDz,tmp_t0,lens,new=False)
        vt = numpy.where( SDF < lens, u0_smooth, 0.0 )
        # Add velocity linear profile in tangential and normal directions
        v1 = v1 + vt*u1 + vn*gDx
        v2 = v2 + vt*u2 + vn*gDy
        v3 = v3 + vt*u3 + vn*gDz

        if phivar:
            v1 += v1_phi
            v2 += v2_phi
            v3 += v3_phi

        return [v1, v2, v3]

    def ibmVel(self,vel,phi,gphi,phivar=None):

        u = vel[0]
        v = vel[1]
        w = vel[2]
    
        phix = gphi[0]
        phiy = gphi[1]
        phiz = gphi[2]

        lens = self.pyranda.mesh.GridLen * immersed_EPS
    
        output =  self.slip_velocity( phi ,phix,phiy,phiz,
                                   u,v,w,lens,new=False,phivar=phivar,slip=True)
        if numpy.linalg.norm(self.pyranda.PyMPI.chunk_3d_lo-numpy.array([0,60,0])) == 0.:
            import pdb; pdb.set_trace()
            print('here we are')
        return output

    def ibmWall(self,vel,phi,gphi,phivar=None):

        u = vel[0]
        v = vel[1]
        w = vel[2]
    
        phix = gphi[0]
        phiy = gphi[1]
        phiz = gphi[2]

        lens = self.pyranda.mesh.GridLen * immersed_EPS
    
        return self.slip_velocity( phi ,phix,phiy,phiz,
                                   u,v,w,lens,phivar=None,slip=False)
        
    
    def ibmS(self,scalar,phi,gphi):
        
        phix = gphi[0]
        phiy = gphi[1]
        phiz = gphi[2]
        
        epsi = 0.0    
    
        return self.smooth_terrain( phi, phix, phiy, phiz,
                                    scalar,epsi)
    
        

    def smooth_terrain(self,SDF,gDx,gDy,gDz,val_in,epsi,new=False):
        
        val = val_in * 1.0
        GridLen = self.pyranda.mesh.GridLen
        
        for i in range(immersed_iter):
            [tvx,tvy,tvz] = self.pyranda.grad(val)
            term = tvx*gDx+tvy*gDy+tvz*gDz
            #term += self.pyranda.laplacian(SDF)*val
            val = numpy.where( SDF <= epsi , val + immersed_CFL*GridLen*term , val )
            Tval = self.pyranda.gfilter(val)
            val = numpy.where( SDF <= epsi , Tval, val )
        
        return val


    def slip_velocity(self,SDF,gDx,gDy,gDz,v1_in,v2_in,v3_in,lens,new=False,phivar=None,slip=True):

        v1 = v1_in*1.0
        v2 = v2_in*1.0
        v3 = v3_in*1.0

        if phivar:
            v1_phi = phivar[0]
            v2_phi = phivar[1]
            v3_phi = phivar[2]

            # Transform to interface velocity
            v1 -= v1_phi
            v2 -= v2_phi
            v3 -= v3_phi

        v1 = self.smooth_terrain(SDF,gDx,gDy,gDz,v1,0.0,new=new)
        v2 = self.smooth_terrain(SDF,gDx,gDy,gDz,v2,0.0,new=new)
        v3 = self.smooth_terrain(SDF,gDx,gDy,gDz,v3,0.0,new=new)
            

                
        if slip:


            norm = v1*gDx+v2*gDy+v3*gDz
            
            vn =  numpy.where( SDF < lens, norm, 0.0 )
            
            # Remove normal velocity
            v1 = v1 - vn*gDx
            v2 = v2 - vn*gDy
            v3 = v3 - vn*gDz
            
            # Compute linear velocity through zero level
            tmp = numpy.where( SDF < lens, 0.0 , norm/(SDF) )            
            tmp = self.smooth_terrain(SDF,gDx,gDy,gDz,tmp,lens,new=new)
            vn = numpy.where( SDF < lens, tmp*SDF, 0.0 )
        
            # Add velocity linear profile
            v1 = v1 + vn*gDx
            v2 = v2 + vn*gDy
            v3 = v3 + vn*gDz

            # Based on reconstructing a velocity gradient across the interface
            #dudn = Cf Re U / L  #  ~ \tau_w / mu  ... Cf = .004, L = Re scale, u\infty
            #dudn = .008 * 1.0e5 * 4.0 / 2.0             
            #norm   = numpy.sqrt( v1*v1+v2*v2+v3*v3 ) + 1.0e-10
            #factor = (1.0 + SDF*dudn/norm)  # 0.9
            if False:

                # Reduce magnitude inside... keep vector
                # ... This is wrong and should scale with SDF somehow...
                v1 = numpy.where( SDF < lens, v1*.9, v1)
                v2 = numpy.where( SDF < lens, v2*.9, v2)
                v3 = numpy.where( SDF < lens, v3*.9, v3)
                
                factor  = (1.0 + SDF*.0001)
            
                v1 = numpy.where( SDF < lens, v1*factor, v1)
                v2 = numpy.where( SDF < lens, v2*factor, v2)
                v3 = numpy.where( SDF < lens, v3*factor, v3)


            
            
            

            
        else:
            norm = numpy.sqrt( v1*v1+v2*v2+v3*v3 )
            
            vn =  numpy.where( SDF < lens, norm, 0.0 )
            tmp = numpy.where( SDF < lens, 0.0 , norm/(SDF) )            

            # temp vectors
            inorm = 1.0 / (norm + 1.0e-16)
            tmpx = v1 * inorm
            tmpy = v2 * inorm
            tmpz = v3 * inorm
            
            v1 -= vn*tmpx
            v2 -= vn*tmpy
            v3 -= vn*tmpz

            #if self.pyranda.PyMPI.master: import pdb; pdb.set_trace()

            # Compute linear velocity through zero level
            tmp = self.smooth_terrain(SDF,gDx,gDy,gDz,tmp,lens,new=new)
            vn = numpy.where( SDF < lens, tmp*SDF, 0.0 )
            
            # Add velocity linear profile
            v1 += vn*tmpx
            v2 += vn*tmpy
            v3 += vn*tmpz
            
            
        if phivar:
            v1 += v1_phi
            v2 += v2_phi
            v3 += v3_phi

        return [v1,v2,v3]

    
