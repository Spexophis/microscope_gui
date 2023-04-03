import tifffile as tf
import numpy as np
import matplotlib.pyplot as P
import csv

fft2 = np.fft.fft2
fftshift = np.fft.fftshift

FLAT_FILE = r'C:/Users/Testa4/Documents/PythonProjects/monalisa/20230120_171517_flatfile.csv'

infl_fcn_file = r'C:/Users/Testa4/Documents/PythonProjects/monalisa/20230120_171517_influence_function.tif'


class aotool():

    def __init__(self):
        self.r = 14
        self.d = 28
        self.a = 0.05
        self.zn = 37
        self.results = []
        self.flatfile()
        self.influencefunction()
        self.zarr, self.zern = self.zernikemodes()
        self.mod = np.arange(self.zn)
        self.zmv = np.zeros(self.zn)
        
    def __del__(self):
        pass

    def influencefunction(self):
        infl_fcn = tf.imread(infl_fcn_file)
        self.Sm = Smat(infl_fcn, radius=self.r)

    def read_cmd(self, fnd):
        cmd = []
        with open(fnd, mode ='r')as file:
          csvFile = csv.reader(file)
          for value in csvFile:
              cmd.append(value)
        return [float(i) for i in cmd[0]]
    
    def flatfile(self):
        # with open(FLAT_FILE, "r") as file:
        #     self.cmd_best = eval(file.readline())
        self.cmd_best = self.read_cmd(FLAT_FILE)
        # self.cmd_best = [0.] * 97
        self.cmd_flat = self.cmd_best

    def zernikemodes(self):
        zern = np.zeros((self.zn, self.d, self.d))
        zarr = np.zeros((self.zn, self.d, self.d))
        for ii in range(self.zn):
            zarr[ii] = self.zernike_noll(ii + 1, self.d)
        # orthogonalize
        for ii in range(self.zn):
            zern[ii] = zarr[ii]
            for jj in range(ii):
                h = (zarr[ii]*zern[jj]).sum()
                zern[ii] = zern[ii] - h*zern[jj]
            zern[ii] = zern[ii]/np.sqrt((zern[ii]**2).sum())
        return zarr, zern

    def testSm(self, mode, amp):
        S = self.Sm.S
        phiin = amp*self.zern[mode]
        dmarr = self.a*np.dot(S,phiin.reshape(self.d*self.d))
        phiout = np.dot(self.Sm.R, dmarr).reshape(self.d,self.d)
        return phiout        
    
    def dmzm(self, mode, amp):
        phiin = amp * self.zern[mode]
        dmarr = self.a * np.dot(self.Sm.S, phiin.reshape((self.d*self.d)))
        # cmd = [0.] * 97
        for i in range(97):
            self.cmd_best[i] = dmarr[i] + self.cmd_best[i]
        if (all(i <= 1.0 for i in self.cmd_best)):
            return self.cmd_best
        else:
            raise Exception(' Error: push value greater than 1.0 ')
    
    def get_zernike(self, mode, amp):
        phiin = amp * self.zern[mode]
        return self.a * np.dot(self.Sm.S, phiin.reshape((self.d*self.d)))
    
    def decomwf(self, wf):
        nx, ny = wf.shape
        self.a = np.zeros(self.zn)
        for i in range(self.zn):
            wz = self.zern[i]
            self.a[i] = ( wf * wz.conj() ).sum() / ( wz * wz.conj() ).sum()
    
    def recomwf(self, d):
        n = self.a.shape[0]
        self.wf = np.zeros((d,d))
        for i in range(n):
            self.wf += self.a[i] * self.zern[i]
    
    def circle(self, radius, size, circle_centre=(0, 0), origin="middle"):
        C = np.zeros((size, size))
        coords = np.arange(0.5, size, 1.0)
        if len(coords) != size:
            raise ("len(coords) = {0}, ".format(len(coords)) +  "size = {0}. They must be equal.".format(size) + "\n Debug the line \"coords = ...\".")
        x, y = np.meshgrid(coords, coords)
        if origin == "middle":
            x -= size / 2.
            y -= size / 2.
        x -= circle_centre[0]
        y -= circle_centre[1]
        mask = x * x + y * y <= radius * radius
        C[mask] = 1
        return C
    
    def phaseFromZernikes(self, zCoeffs, size, norm="noll"):
        """
        Creates an array of the sum of zernike polynomials with specified coefficeints
        Parameters:
            zCoeffs (list): zernike Coefficients
            size (int): Diameter of returned array
            norm (string, optional): The normalisation of Zernike modes. Can be ``"noll"``, ``"p2v"`` (peak to valley), or ``"rms"``. default is ``"noll"``.
        Returns:
            ndarray: a `size` x `size` array of summed Zernike polynomials
        """
        Zs = self.zernikeArray(len(zCoeffs), size, norm=norm)
        phase = np.zeros((size, size))
        for z in range(len(zCoeffs)):
            phase += Zs[z] * zCoeffs[z]
        return phase
    
    def zernike_noll(self, j, N):
        """
         Creates the Zernike polynomial with mode index j,
         where j = 1 corresponds to piston.
         Args:
            j (int): The noll j number of the zernike mode
            N (int): The diameter of the zernike more in pixels
         Returns:
            ndarray: The Zernike mode
         """
        n, m = self.zernIndex(j)
        return self.zernike_nm(n, m, N)
    
    def zernike_nm(self, n, m, N):
        """
         Creates the Zernike polynomial with radial index, n, and azimuthal index, m.
         Args:
            n (int): The radial order of the zernike mode
            m (int): The azimuthal order of the zernike mode
            N (int): The diameter of the zernike more in pixels
         Returns:
            ndarray: The Zernike mode
         """
        coords = (np.arange(N) - N / 2. + 0.5) / (N / 2.)
        X, Y = np.meshgrid(coords, coords)
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
    
        if m==0:
            Z = np.sqrt(n+1) * self.zernikeRadialFunc(n, 0, R)
        else:
            if m > 0: # j is even
                Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.cos(m*theta)
            else:   #i is odd
                m = abs(m)
                Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.sin(m * theta)
        Z = Z*np.less_equal(R, 1.0)
        return Z*self.circle(N/2., N)
    
    def zernikeRadialFunc(self, n, m, r):
        """
        Fucntion to calculate the Zernike radial function
        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the centre the array
        Returns:
            ndarray: The Zernike radial function
        """
        R = np.zeros(r.shape)
        for i in range(0, int((n - m) / 2) + 1):
            R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                             np.math.factorial(n - i)) /
                             (np.math.factorial(i) *
                              np.math.factorial(0.5 * (n + m) - i) *
                              np.math.factorial(0.5 * (n - m) - i)),
                             dtype='float')
        return R
    
    def zernIndex(self, j):
        """
        Find the [n,m] list giving the radial order n and azimuthal order
        of the Zernike polynomial of Noll index j.
        Parameters:
            j (int): The Noll index for Zernike polynomials
        Returns:
            list: n, m values
        """
        n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n%2
        m = int((p+k)/2.)*2 - k
        if m!=0:
            if j%2==0:
                s=1
            else:
                s=-1
            m *= s
        return [n, m]
    
    def zernikeArray(self, J, N, norm="noll"):
        """
        Creates an array of Zernike Polynomials
        Parameters:
            maxJ (int or list): Max Zernike polynomial to create, or list of zernikes J indices to create
            N (int): size of created arrays
            norm (string, optional): The normalisation of Zernike modes. Can be ``"noll"``, ``"p2v"`` (peak to valley), or ``"rms"``. default is ``"noll"``.
        Returns:
            ndarray: array of Zernike Polynomials
        """
        # If list, make those Zernikes
        try:
            nJ = len(J)
            Zs = np.empty((nJ, N, N))
            for i in range(nJ):
                Zs[i] = self.zernike_noll(J[i], N)
        # Else, cast to int and create up to that number
        except TypeError:
            maxJ = int(np.round(J))
            N = int(np.round(N))
            Zs = np.empty((maxJ, N, N))
            for j in range(1, maxJ+1):
                Zs[j-1] = self.zernike_noll(j, N)
        if norm=="p2v":
            for z in range(len(Zs)):
                Zs[z] /= (Zs[z].max()-Zs[z].min())
        elif norm=="rms":
            for z in range(len(Zs)):
                # Norm by RMS. Remember only to include circle elements in mean
                Zs[z] /= np.sqrt(
                        np.sum(Zs[z]**2)/np.sum(self.circle(N/2., N)))
        return Zs
    
    def makegammas(self, nzrad):
        """
        Make "Gamma" matrices which can be used to determine first derivative
        of Zernike matrices (Noll 1976).
        Parameters:
            nzrad: Number of Zernike radial orders to calculate Gamma matrices for
        Return:
            ndarray: Array with x, then y gamma matrices
        """
        n=[0]
        m=[0]
        tt=[1]
        trig=0
        for p in range(1,nzrad+1):
            for q in range(p+1):
                if(np.fmod(p-q,2)==0):
                    if(q>0):
                        n.append(p)
                        m.append(q)
                        trig=not(trig)
                        tt.append(trig)
                        n.append(p)
                        m.append(q)
                        trig=not(trig)
                        tt.append(trig)
                    else:
                        n.append(p)
                        m.append(q)
                        tt.append(1)
                        trig=not(trig)
        nzmax=len(n)
        #for j in range(nzmax):
            #print j+1, n[j], m[j], tt[j]
        gamx = np.zeros((nzmax,nzmax),"float32")
        gamy = np.zeros((nzmax,nzmax),"float32")
        # Gamma x
        for i in range(nzmax):
            for j in range(i+1):
                # Rule a:
                if (m[i]==0 or m[j]==0):
                    gamx[i,j] = np.sqrt(2.0)*np.sqrt(float(n[i]+1)*float(n[j]+1))
                else:
                    gamx[i,j] = np.sqrt(float(n[i]+1)*float(n[j]+1))
                # Rule b:
                if m[i]==0:
                    if ((j+1) % 2) == 1:
                        gamx[i,j] = 0.0
                elif m[j]==0:
                    if ((i+1) % 2) == 1:
                        gamx[i,j] = 0.0
                else:
                    if ( ((i+1) % 2) != ((j+1) % 2) ):
                        gamx[i,j] = 0.0
                # Rule c:
                if abs(m[j]-m[i]) != 1:
                    gamx[i,j] = 0.0
                # Rule d - all elements positive therefore already true
        # Gamma y
        for i in range(nzmax):
            for j in range(i+1):
                # Rule a:
                if (m[i]==0 or m[j]==0):
                    gamy[i,j] = np.sqrt(2.0)*np.sqrt(float(n[i]+1)*float(n[j]+1))
                else:
                    gamy[i,j] = np.sqrt(float(n[i]+1)*float(n[j]+1))
                # Rule b:
                if m[i]==0:
                    if ((j+1) % 2) == 0:
                        gamy[i,j] = 0.0
                elif m[j]==0:
                    if ((i+1) % 2) == 0:
                        gamy[i,j] = 0.0
                else:
                    if ( ((i+1) % 2) == ((j+1) % 2) ):
                        gamy[i,j] = 0.0
                # Rule c:
                if abs(m[j]-m[i]) != 1:
                    gamy[i,j] = 0.0
                # Rule d:
                if m[i]==0:
                    pass    # line 1
                elif m[j]==0:
                    pass    # line 1
                elif m[j]==(m[i]+1):
                    if ((i+1) % 2) == 1:
                        gamy[i,j] *= -1.    # line 2
                elif m[j]==(m[i]-1):
                    if ((i+1) % 2) == 0:
                        gamy[i,j] *= -1.    # line 3
                else:
                    pass    # line 4
        return np.array([gamx,gamy])


class Smat(object):

    def __init__(self, stack, radius=None):
        self.wfstack = stack.copy()
        self.nz, nx, ny = self.wfstack.shape
        if not(self.nz == 97):
            raise 'There must be 97 images!'
        # set size of array
        if radius==None:
            radius = 15
        self.sz= 2*radius
        # cut out wavefront
#        R0 = self.wfstack[:,(nx/2-radius):(nx/2+radius),(nx/2-radius):(nx/2+radius)]
        R0 = self.wfstack
        msk = (R0[0]!=0.0).astype(np.float32)
        nz, nx, ny = R0.shape
        # remove mean
        for m in range(nz):
            mn = R0[m].sum()/msk.sum()
            R0[m] = msk*(R0[m]-mn)
        R = np.zeros((nx*ny,nz))
        for m in range(nz):
            R[:,m] = R0[m].reshape(nx*ny)
        self.R = R
        self.calcS(21)
        #Y.view(R0)

    def __del__(self):
        pass

    def calcS(self, Ns=21): # SVD decomposition of R. Use largest Ns vectors.
        ''' Ns is number of singular values to retain '''
        u,s,vh = np.linalg.svd(self.R,0,1)
        ut = np.transpose(u)
        self.s = s
        sd = 1./s
        s = np.zeros((self.nz, self.nz))
        for i in range(Ns):
            s[i,i] = sd[i]
        v = np.transpose(vh)
        t = np.dot(v,s)
        self.u = u
        self.v = v
        self.S = np.dot(t,ut)
        return True

    def calcSalpha(self,Ns=21,alpha=2.0): # SVD decomposition of R. Use largest Ns vectors.
        ''' Ns is number of singular values to retain '''
        u,s,vh = np.linalg.svd(self.R,0,1)
        ut = np.transpose(u)
        self.s = s
        sd = 1./np.sqrt(s**2 + alpha**2)
        s = np.zeros((self.nz, self.nz))
        for i in range(Ns):
            s[i,i] = sd[i]
        v = np.transpose(vh)
        t = np.dot(v,s)
        self.u = u
        self.v = v
        self.S = np.dot(t,ut)
        return True

    def view_eigen_vectors(self):
        t = (self.u.reshape(self.sz,self.sz, self.nz)).swapaxes(1,2).swapaxes(0,1)
        tf.imshow(t,vmin=None)
        P.plot(self.s)
        return True
