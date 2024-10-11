"""A place for modes that aren't in prysm.polynomials"""
import scipy
from prysm.mathops import np

def hadamard_modes_sequence(aperture):
    """Generate a sequence of hadamard modes

    Code adapted from the uasal/lina package
    https://github.com/uasal/lina/blob/main/lina/utils.py

    Parameters
    ----------
    aperture : ndarray
        binary array denoting the aperture transmission function

    Returns
    -------
    list of ndarrays
        sequence of hadamard modes
    """
        
    num_actuators = aperture.sum().astype(int)
    shape_actuators = aperture.shape[0]
    np2 = 2**int(np.ceil(np.log2(num_actuators)))
    hmodes = np.array(scipy.linalg.hadamard(np2))
    
    had_modes = []

    inds = np.where(aperture.flatten().astype(int))

    for hmode in hmodes:
        hmode = hmode[:num_actuators]
        mode = np.zeros((aperture.shape[0]**2))
        mode[inds] = hmode
        had_modes.append(mode)

    had_modes = np.array(had_modes).reshape(np2, shape_actuators, shape_actuators)
    
    return had_modes
