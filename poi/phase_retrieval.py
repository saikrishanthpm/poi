from prysm.mathops import np

from prysm.propagation import (
    focus_fixed_sampling,
    focus_fixed_sampling_backprop,
    Wavefront
)

from prysm.coordinates import (
    make_xy_grid,
    cart_to_polar
)

from prysm.polynomials import (
    noll_to_nm,
    zernike_nm,
    zernike_nm_sequence,
    sum_of_2d_modes
)

from prysm.thinlens import image_displacement_to_defocus

class _ADPhaseRetrieval:

    def __init__(self, amp, target, IC, basis, defocus=0, initial_phase=None):
        """init an ADPR Experiment for a given wavelength and defocus position

        Parameters
        ----------
        amp : numpy.ndarray
            pupil amplitude mask
        target : numpy.ndarray
            measured point-spread function
        IC : dict of floats
            dictionary containing the following keywords, ex:
            'image_dx' : 4.87, # microns
            'pupil_diameter' : 10950, # mm
            'focal_length' : 338492, # mm
            'wavelength' : 0.94, # microns
        defocus : float, optional
            defocus distance of the target in mm, by default 0 mm
        initial_phase : numpy.ndarray, optional
            initial phase estimate in the pupil, by default None
        """

        if initial_phase is None:
            phs = np.zeros(amp.shape)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.phs = phs
        self.D = target
        self.defocus_distance = defocus
        self.efl = IC['focal_length']
        self.dpup = IC['pupil_diameter']
        self.img_dx = IC['image_dx']
        self.wvl = IC['wavelength']
        self.basis = basis
        self.fno = self.efl / self.dpup
        
        # compute number of samples
        self.npup = self.amp.shape[0]
        self.nimg = self.D.shape[0]

        # compute the pupil dx
        self.pup_dx = self.dpup / self.npup

        # configure the defocus applied
        # self.defocus_microns = image_displacement_to_defocus(self.defocus_distance, self.fno)
        # x, y = make_xy_grid(self.npup, diameter=1)
        # r, t = cart_to_polar(x, y)
        # defocus_polynomial = zernike_nm(2, 0, r, t)
        # defocus_polynomial /= np.max(np.abs(defocus_polynomial))
        # defocus_polynomial *= self.defocus_microns
        # self.defocus_rad = (4 * np.pi / self.wvl) * defocus_polynomial
        self.cost = []
        self.set_optimization_method()

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0, 0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = phs
        self.wf = Wavefront.from_amp_and_phase(self.amp, W, self.wvl, self.pup_dx)
        self.g = self.wf.data
        self.wf = self.wf.focus_fixed_sampling(self.efl, self.img_dx, self.nimg)
        self.wf = self.wf.free_space(self.defocus_distance * 1e3)
        G = self.wf.data
        I = np.abs(G)**2
        E = np.sum((I - self.D)**2)

        # save things
        self.phs = phs
        self.W = W
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        self.cost.append(self.E)
        return self.E
    
    def rev(self, x):
        self.update(x)

        Ibar = 2 * (self.I - self.D)
        Gbar = 2 * Ibar * self.G
        self.wfbar = Wavefront(Gbar, self.wvl, self.img_dx, space='psf')
        self.gbar = self.wfbar.data
        self.wfbar = self.wfbar.free_space(-self.defocus_distance * 1e3)
        self.wfbar = self.wfbar.focus_fixed_sampling_backprop(self.efl, self.pup_dx, self.npup)
        Wbar = self.wfbar.from_amp_and_phase_backprop_phase(self.wfbar.data)

        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]
        

class ADPhaseRetrieval:

    def __init__(self, amp, target, IC, basis, defocus=0, initial_phase=None):
        """init an ADPR Experiment for a given wavelength and defocus position

        Parameters
        ----------
        amp : numpy.ndarray
            pupil amplitude mask
        target : numpy.ndarray
            measured point-spread function
        IC : dict of floats
            dictionary containing the following keywords, ex:
            'image_dx' : 4.87, # microns
            'pupil_diameter' : 10950, # mm
            'focal_length' : 338492, # mm
            'wavelength' : 0.94, # microns
        defocus : float, optional
            defocus distance of the target in mm, by default 0 mm
        initial_phase : numpy.ndarray, optional
            initial phase estimate in the pupil, by default None
        """

        if initial_phase is None:
            phs = np.zeros(amp.shape)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.phs = phs
        self.D = target
        self.defocus_distance = defocus
        self.efl = IC['focal_length']
        self.dpup = IC['pupil_diameter']
        self.img_dx = IC['image_dx']
        self.wvl = IC['wavelength']
        self.basis = basis
        self.fno = self.efl / self.dpup
        
        # compute number of samples
        self.npup = self.amp.shape[0]
        self.nimg = self.D.shape[0]

        # compute the pupil dx
        self.pup_dx = self.dpup / self.npup

        # configure the defocus applied
        self.defocus_microns = image_displacement_to_defocus(self.defocus_distance, self.fno)
        x, y = make_xy_grid(self.npup, diameter=1)
        r, t = cart_to_polar(x, y)
        defocus_polynomial = zernike_nm(2, 0, r, t)
        defocus_polynomial /= np.max(np.abs(defocus_polynomial))
        defocus_polynomial *= self.defocus_microns
        self.defocus_rad = (4 * np.pi / self.wvl) * defocus_polynomial
        self.cost = []
        self.set_optimization_method()

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0, 0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs / 1e3
        W -= self.defocus_rad
        g = self.amp * np.exp(1j * W)
        G = focus_fixed_sampling(
            wavefunction=g,
            input_dx=self.pup_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.img_dx,
            output_samples=self.nimg,
            shift=(0, 0),
            method='mdft')
        I = np.abs(G)**2
        E = np.sum((I - self.D)**2)

        # save things
        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        self.cost.append(self.E)
        return self.E
    
    def rev(self, x):
        self.update(x)

        Ibar = 2 * (self.I - self.D)
        Gbar = 2 * Ibar * self.G
        gbar = focus_fixed_sampling_backprop(
                wavefunction=Gbar,
                input_dx=self.pup_dx,
                prop_dist = self.efl,
                wavelength=self.wvl,
                output_dx=self.img_dx,
                output_samples=self.phs.shape,
                shift=(0, 0),
                method='mdft')
        
        Wbar = 2 * np.pi / self.wvl * np.imag(gbar * np.conj(self.g)) / 1e3

        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

class _ADPhaseRetrieval:

    def __init__(self, amp, target, IC, basis, defocus=0, initial_phase=None):
        """init an ADPR Experiment for a given wavelength and defocus position

        Parameters
        ----------
        amp : numpy.ndarray
            pupil amplitude mask
        target : numpy.ndarray
            measured point-spread function
        IC : dict of floats
            dictionary containing the following keywords, ex:
            'image_dx' : 4.87, # microns
            'pupil_diameter' : 10950, # mm
            'focal_length' : 338492, # mm
            'wavelength' : 0.94, # microns
        defocus : float, optional
            defocus distance of the target in mm, by default 0 mm
        initial_phase : numpy.ndarray, optional
            initial phase estimate in the pupil, by default None
        """

        if initial_phase is None:
            phs = np.zeros(amp.shape)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.phs = phs
        self.D = target
        self.defocus_distance = defocus
        self.efl = IC['focal_length']
        self.dpup = IC['pupil_diameter']
        self.img_dx = IC['image_dx']
        self.wvl = IC['wavelength']
        self.basis = basis
        self.fno = self.efl / self.dpup
        
        # compute number of samples
        self.npup = self.amp.shape[0]
        self.nimg = self.D.shape[0]

        # compute the pupil dx
        self.pup_dx = self.dpup / self.npup

        # configure the defocus applied
        self.defocus_microns = image_displacement_to_defocus(self.defocus_distance, self.fno)
        x, y = make_xy_grid(self.npup, diameter=1)
        r, t = cart_to_polar(x, y)
        defocus_polynomial = zernike_nm(2, 0, r, t)
        defocus_polynomial /= np.max(np.abs(defocus_polynomial))
        defocus_polynomial *= self.defocus_microns
        self.defocus_rad = (4 * np.pi / self.wvl) * defocus_polynomial
        self.cost = []
        self.set_optimization_method()

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0, 0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x
               
        # configure wavefront
        wf = Wavefront.from_amp_and_phase(self.amp,
                                          self.phs,
                                          self.wvl,
                                          self.pup_dx)
        self.W = np.angle(wf.data)
        self.g = wf.data
        wf = wf.focus_fixed_sampling(efl=self.efl,
                                     dx=self.img_dx,
                                     samples=self.nimg)
        I = self.wf.intensity
        E = np.sum((I - self.D)**2)

        # save things
        self.phs = phs
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        self.cost.append(self.E)
        return self.E
    
    def rev(self, x):
        self.update(x)

        Ibar = 2 * (self.I - self.D)
        Gbar = 2 * Ibar * self.G
        wf = Gbar.focus_fixed_sampling_backprop(efl=self.efl,
                                              dx=self.pup_dx,
                                              samples=self.npup)
        Wbar = self.wf.from_amp_and_phase_backprop_phase(wf.data)

        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

"""Taken from dydgug.vappid.VAPPOptimizer2"""
class VAPPOptimizer2:
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx, defocus_waves=0, initial_phase=None):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh_dx = dh_dx
        self.D = dark_hole
        self.phs = phs
        self.zonal = False
        self.defocus = defocus_waves

        # configure the defocus polynomial
        x, y = make_xy_grid(amp.shape[0], diameter=2)
        r, t = cart_to_polar(x, y)
        self.defocus_polynomial = 2 * zernike_nm(2, 0, r, t)
        self.defocus_polynomial /= np.max(np.abs(self.defocus_polynomial))
        self.defocus_aberration = 2 * np.pi * self.defocus_polynomial * self.defocus * self.amp
        self.cost = []

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0,0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs

        # TODO: Check if this is a minus sign instead
        W -= self.defocus_aberration
        g = self.amp * np.exp(1j * W)
        G = focus_fixed_sampling(
            wavefunction=g,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.D.shape,
            shift=(0, 0),
            method='mdft')
        I = np.abs(G)**2
        E = np.sum((I - self.D)**2)
        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = 2*(self.I - self.D)
        Gbar = 2 * Ibar * self.G
        gbar = focus_fixed_sampling_backprop(
            wavefunction=Gbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.phs.shape,
            shift=(0, 0),
            method='mdft')

        Wbar = 2 * np.pi / self.wvl * np.imag(gbar * np.conj(self.g))
        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        self.cost.append(f)
        return f, g

class ParallelADPhaseRetrieval:

    def __init__(self, optlist):

        self.optlist = optlist
        self.f = 0
        self.g = 0
        self.cost = []

    def refresh(self):
        self.f = 0
        self.g = 0
    
    def fg(self, x):

        # reset the f, g values
        self.refresh()

        # just sum them
        for opt in self.optlist:
            f, g = opt.fg(x)
            self.f += f
            self.g += g
        
        self.cost.append(self.f)

        return self.f, self.g

