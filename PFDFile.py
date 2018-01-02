# Standard library Imports:
import struct, sys

# Numpy Imports:
from numpy import array
from numpy import asarray
from numpy import concatenate
from numpy import floor
from numpy import fabs
from numpy import fromfile
from numpy import reshape
from numpy import float64
from numpy import arange
from numpy import add
from numpy import mean
from numpy import zeros
from numpy import shape
from numpy import sum
from numpy import sqrt
from numpy import std
import numpy as np
# For plotting fits etc.
# import matplotlib.pyplot as plt

# Custom file Imports:
from PFDFeatureExtractor import PFDFeatureExtractor


class PFD():
    """
    Represents an individual pulsar candidate.

    """
    def __init__(self,candidateName):
        """
        Default constructor.

        Parameters:
        candidateName -    the name for the candidate, typically the file path.
        """
        self.cand = candidateName
        self.features = []
        self.fe = PFDFeatureExtractor(False)
        self.load()

    # ****************************************************************************************************

    def load(self):
        """
        Attempts to load candidate data from the file, performs file consistency checks if the
        debug flag is set to true. Much of this code has been extracted from PRESTO by Scott Ransom.

        Please see:

        http://www.cv.nrao.edu/~sransom/presto/
        https://github.com/scottransom/presto

        Parameters:
        N/A

        Return:
        N/A
        """
        infile = open(self.cand, "rb")

        # The code below appears to have been taken from Presto. So it maybe
        # helpful to look at the Presto github repository (see above) to get a better feel
        # for what this code is doing. I certainly have no idea what is going on. Although
        # data is being unpacked in a specific order.

        swapchar = '<' # this is little-endian
        data = infile.read(5*4)
        testswap = struct.unpack(swapchar+"i"*5, data)
        # This is a hack to try and test the endianness of the data.
        # None of the 5 values should be a large positive number.

        if (fabs(asarray(testswap))).max() > 100000:
            swapchar = '>' # this is big-endian

        (self.numdms, self.numperiods, self.numpdots, self.nsub, self.npart) = struct.unpack(swapchar+"i"*5, data)
        (self.proflen, self.numchan, self.pstep, self.pdstep, self.dmstep, self.ndmfact, self.npfact) = struct.unpack(swapchar+"i"*7, infile.read(7*4))
        self.filenm = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.candnm = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.telescope = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.pgdev = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])

        test = infile.read(16)
        has_posn = 1
        test = test.decode()
        for ii in range(16):
            if test[ii] not in '0123456789:.-\0':
                has_posn = 0
                break

        if has_posn:
            self.rastr = test[:test.find('\0')]
            # self.rastr = test
            test = infile.read(16)
            test = test.decode()
            self.decstr = test[:test.find('\0')]
            # self.decstr = test

            (self.dt, self.startT) = struct.unpack(swapchar+"dd", infile.read(2*8))

        else:
            self.rastr = "Unknown"
            self.decstr = "Unknown"
            (self.dt, self.startT) = struct.unpack(swapchar+"dd", test)

        (self.endT, self.tepoch, self.bepoch, self.avgvoverc, self.lofreq,self.chan_wid, self.bestdm) = struct.unpack(swapchar+"d"*7, infile.read(7*8))
        (self.topo_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.topo_p1, self.topo_p2, self.topo_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.bary_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.bary_p1, self.bary_p2, self.bary_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.fold_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.fold_p1, self.fold_p2, self.fold_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.orb_p, self.orb_e, self.orb_x, self.orb_w, self.orb_t, self.orb_pd,self.orb_wd) = struct.unpack(swapchar+"d"*7, infile.read(7*8))
        self.dms = asarray(struct.unpack(swapchar+"d"*self.numdms,infile.read(self.numdms*8)))

        if self.numdms==1:
            self.dms = self.dms[0]

        self.periods = asarray(struct.unpack(swapchar + "d" * self.numperiods,infile.read(self.numperiods*8)))
        self.pdots = asarray(struct.unpack(swapchar + "d" * self.numpdots,infile.read(self.numpdots*8)))
        self.numprofs = self.nsub * self.npart

        if (swapchar=='<'):  # little endian
            self.profs = zeros((self.npart, self.nsub, self.proflen), dtype='d')
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    try:
                        self.profs[ii,jj,:] = fromfile(infile, float64, self.proflen)
                    except Exception: # Catch *all* exceptions.
                        pass
                        #print ""
        else:
            self.profs = asarray(struct.unpack(swapchar+"d"*self.numprofs*self.proflen,infile.read(self.numprofs*self.proflen*8)))
            self.profs = reshape(self.profs, (self.npart, self.nsub, self.proflen))

        self.binspersec = self.fold_p1 * self.proflen
        self.chanpersub = self.numchan / self.nsub
        self.subdeltafreq = self.chan_wid * self.chanpersub
        self.hifreq = self.lofreq + (self.numchan-1) * self.chan_wid
        self.losubfreq = self.lofreq + self.subdeltafreq - self.chan_wid
        self.subfreqs = arange(self.nsub, dtype='d')*self.subdeltafreq + self.losubfreq
        self.subdelays_bins = zeros(self.nsub, dtype='d')
        self.killed_subbands = []
        self.killed_intervals = []
        self.pts_per_fold = []

        # Note: a foldstats struct is read in as a group of 7 doubles
        # the correspond to, in order:
        # numdata, data_avg, data_var, numprof, prof_avg, prof_var, redchi
        self.stats = zeros((self.npart, self.nsub, 7), dtype='d')

        for ii in range(self.npart):
            currentstats = self.stats[ii]

            for jj in range(self.nsub):
                if (swapchar=='<'):  # little endian
                    try:
                        currentstats[jj] = fromfile(infile, float64, 7)
                    except Exception: # Catch *all* exceptions.
                        pass
                        #print ""
                else:
                    try:
                        currentstats[jj] = asarray(struct.unpack(swapchar+"d"*7,infile.read(7*8)))
                    except Exception: # Catch *all* exceptions.
                        pass
                        #print ""

            self.pts_per_fold.append(self.stats[ii][0][0])  # numdata from foldstats

        self.start_secs = add.accumulate([0]+self.pts_per_fold[:-1])*self.dt
        self.pts_per_fold = asarray(self.pts_per_fold)
        self.mid_secs = self.start_secs + 0.5*self.dt*self.pts_per_fold

        if (not self.tepoch==0.0):
            self.start_topo_MJDs = self.start_secs/86400.0 + self.tepoch
            self.mid_topo_MJDs = self.mid_secs/86400.0 + self.tepoch

        if (not self.bepoch==0.0):
            self.start_bary_MJDs = self.start_secs/86400.0 + self.bepoch
            self.mid_bary_MJDs = self.mid_secs/86400.0 + self.bepoch

        self.Nfolded = add.reduce(self.pts_per_fold)
        self.T = self.Nfolded*self.dt
        self.avgprof = (self.profs/self.proflen).sum()
        self.varprof = self.calc_varprof()
        self.barysubfreqs = self.subfreqs
        infile.close()

    # ****************************************************************************************************
    def getprofile(self):
        """
        Obtains the profile data from the candidate file.

        Parameters:
        N/A

        Returns:
        The candidate profile data (an array) scaled to within the range [0,255].
        """
        # if not self.__dict__.has_key('subdelays'):
        if 'subdelays' not in self.__dict__:
            self.dedisperse()
        
        self.sumprof = self.profs.sum(0).sum(0) # add
        normprof = self.sumprof - min(self.sumprof)

        s = normprof / mean(normprof)

        return self.scale(s)

    # ****************************************************************************************************

    def scale(self,data,attr = 'profile'):
        """
        Scales the profile data for pfd files so that it is in the range 0-255.
        This is the same range used in the phcx files. So  by performing this scaling
        the features for both type of candidates are directly comparable. Before it was
        harder to determine if the features generated for pfd files were working correctly,
        since the phcx features are our only point of reference.

        Parameter:
        data    -    the data to scale to within the 0-255 range.

        Returns:
        A new array with the data scaled to within the range [0,255].
        """
        min_=min(data)
        max_=max(data)

        newMin=0;
        newMax=255

        newData=[]

        for n in range(len(data)):

            value=data[n]
            x = (newMin * (1-( (value-min_) /( max_-min_ )))) + (newMax * ( (value-min_) /( max_-min_ ) ))
            newData.append(x)
        if attr == 'sub_plot':
            for e_no in range(len(newData)):
                newData[e_no] = newMax - newData[e_no]
        return newData

    # ****************************************************************************************************

    def calc_varprof(self):
        """
        This function calculates the summed profile variance of the current pfd file.
        Killed profiles are ignored. I have no idea what a killed profile is. But it
        sounds fairly gruesome.
        """
        varprof = 0.0
        for part in range(self.npart):
            if part in self.killed_intervals: continue
            for sub in range(self.nsub):
                if sub in self.killed_subbands: continue
                varprof += self.stats[part][sub][5] # foldstats prof_var
        return varprof
        # ****************************************************************************************************

    def dedisperse(self, DM=None, interp=1):
        """
        Rotate (internally) the profiles so that they are de-dispersed
        at a dispersion measure of DM.  Use FFT-based interpolation if
        'interp' is non-zero (NOTE: It is off by default!).
        """

        if DM is None:
            DM = self.bestdm

        # Note:  Since TEMPO pler corrects observing frequencies, for
        #        TOAs, at least, we need to de-disperse using topocentric
        #        observing frequencies.
        self.subdelays = self.fe.delay_from_DM(DM, self.subfreqs)
        self.hifreqdelay = self.subdelays[-1]
        self.subdelays = self.subdelays - self.hifreqdelay
        delaybins = self.subdelays * self.binspersec - self.subdelays_bins

        if interp:

            new_subdelays_bins = delaybins

            for ii in range(self.npart):
                for jj in range(self.nsub):
                    tmp_prof = self.profs[ii, jj, :]
                    self.profs[ii, jj] = self.fe.fft_rotate(tmp_prof, delaybins[jj])

            # Note: Since the rotation process slightly changes the values of the
            # profs, we need to re-calculate the average profile value
            self.avgprof = (self.profs / self.proflen).sum()

        else:

            new_subdelays_bins = floor(delaybins + 0.5)

            for ii in range(self.nsub):

                rotbins = int(new_subdelays_bins[ii]) % self.proflen
                if rotbins:  # i.e. if not zero
                    subdata = self.profs[:, ii, :]
                    self.profs[:, ii] = concatenate((subdata[:, rotbins:], subdata[:, :rotbins]), 1)

        self.subdelays_bins += new_subdelays_bins
        self.sumprof = self.profs.sum(0).sum(0)
    # ****************************************************************************************************

    def plot_chi2_vs_DM(self, loDM, hiDM, N=100, interp=0):
        """
        Plot (and return) an array showing the reduced-chi^2 versus DM
        (N DMs spanning loDM-hiDM). Use sinc_interpolation if 'interp' is non-zero.
        """

        # Sum the profiles in time
        sumprofs = self.profs.sum(0)

        if not interp:
            profs = sumprofs
        else:
            profs = zeros(shape(sumprofs), dtype='d')

        DMs = self.fe.span(loDM, hiDM, N)
        chis = zeros(N, dtype='f')
        subdelays_bins = self.subdelays_bins.copy()

        for ii, DM in enumerate(DMs):

            subdelays = self.fe.delay_from_DM(DM, self.barysubfreqs)
            hifreqdelay = subdelays[-1]
            subdelays = subdelays - hifreqdelay
            delaybins = subdelays*self.binspersec - subdelays_bins

            if interp:

                interp_factor = 16
                for jj in range(self.nsub):
                    profs[jj] = self.fe.interp_rotate(sumprofs[jj], delaybins[jj],zoomfact=interp_factor)
                # Note: Since the interpolation process slightly changes the values of the
                # profs, we need to re-calculate the average profile value
                avgprof = (profs/self.proflen).sum()

            else:

                new_subdelays_bins = floor(delaybins+0.5)
                for jj in range(self.nsub):
                    profs[jj] = self.fe.rotate(profs[jj], int(new_subdelays_bins[jj]))
                subdelays_bins += new_subdelays_bins
                avgprof = self.avgprof

            sumprof = profs.sum(0)
            chis[ii] = self.calc_redchi2(prof=sumprof, avg=avgprof)

        return (chis, DMs)

    # ******************************************************************************************

    def calc_redchi2(self, prof=None, avg=None, var=None):
        """
        Return the calculated reduced-chi^2 of the current summed profile.
        """

        # dict.has_key has been removed in 3.x
        #if not self.__dict__.has_key('subdelays'):
        if not 'subdelays' in self.__dict__:
            self.dedisperse()

        if prof is None:  prof = self.sumprof
        if avg is None:  avg = self.avgprof
        if var is None:  var = self.varprof
        return ((prof-avg)**2.0/var).sum()/(len(prof)-1.0)

    # ******************************************************************************************

    def get_subbands(self,is_scaled = True):
        """
        Plot the interval-summed profiles vs subband.  Restrict the bins
        in the plot to the (low:high) slice defined by the phasebins option
        if it is a tuple (low,high) instead of the string 'All'.
        """
        # if not self.__dict__.has_key('subdelays'):
        if 'subdelays' not in self.__dict__:
            self.dedisperse()

        lo, hi = 0.0, self.proflen
        profs = self.profs.sum(0)
        lof = self.lofreq - 0.5*self.chan_wid
        hif = lof + self.chan_wid*self.numchan
        # scale
        if is_scaled:
            for row_no in range(len(profs)):
                profs[row_no] = self.scale(profs[row_no],attr = 'sub_plot')
        return profs


    def get_subints(self, is_scaled=True):
        """
        Plot the interval-summed profiles vs subband.  Restrict the bins
        in the plot to the (low:high) slice defined by the phasebins option
        if it is a tuple (low,high) instead of the string 'All'.
        """
        # if not self.__dict__.has_key('subdelays'):
        if 'subdelays' not in self.__dict__:
            self.dedisperse()

        lo, hi = 0.0, self.proflen
        profs = self.profs.sum(1)
        lof = self.lofreq - 0.5 * self.chan_wid
        hif = lof + self.chan_wid * self.numchan
        # scale
        if is_scaled:
            for row_no in range(len(profs)):
                profs[row_no] = self.scale(profs[row_no], attr='sub_plot')
        return profs

    def get_profs(self,is_scaled=True):
        if is_scaled:
            data = self.profs
            min_=np.amin(data)
            max_=np.amax(data)
            newMin=0;
            newMax=255
            t_size, f_size, p_size = data.shape
            for t_no in range(t_size):
                for f_no in range(f_size):
                    for p_no in range(p_size):
                        value = data[t_no][f_no][p_no]
                        data[t_no][f_no][p_no] = newMax - ((newMin * (1 - ((value - min_) / (max_ - min_)))) + (newMax * ((value - min_) / (max_ - min_))))
            return data
        else:
            return self.profs
