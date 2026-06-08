import numpy as np

from scipy.sparse.linalg import cg, LinearOperator
from scipy.interpolate import CubicSpline

def _check_lr(left,right):
    '''
    Checks if the arguments left and right are well defined (they must be different)
    '''
    if left == right:
        raise ValueError("Exactly one of left or right must be True")

def _check_Pv(P,v):
    '''
    Checks if the arguments P and v are well defined: len(v) < len(P)
    '''
    if len(v) >= len(P):
        raise ValueError("len(v) must be < len(P)")

def _padding(v,N,left=False,right=False):
    '''
    Given v, returns a padded vector, (0,v) or (v,0) depending on whether left or right are True.
    
    ARGUMENTS____________________________________________ 
    v:     vector to be padded
    N:     final length of the padded vector
    left:  if true, returns (0,v) 
    right: if true, returns (v,0) 
    '''
    _check_lr(left,right)
    if len(v) >= N:
        raise ValueError("len(v) must be < N")
    n = len(v)
    z = np.zeros(N)
    if left:
        z[-n:] = v   
    else: 
        z[:n] = v
    return z

def _A_func(Pinv,v,nx,left=False,right=False):
    '''
    Given v, defines z as (0,v) or (v,0), depending on whether left or right are True, such that z is as long as Pinv.
    Then computes A(z) = IDFT(Pinv * DFT(z)) and returns the first nx elements of A(z)

    ARGUMENTS____________________________________________ 
    Pinv:  inverse of the power spectrum, as a function of frequency
    v:     vector to be padded
    nx:    length of thhe output vector (number of elements of A(z) to be returned)
    left:  if true, returns (0,v) 
    right: if true, returns (v,0) 
    '''
    _check_Pv(Pinv,v)
    _check_lr(left,right)
    n = len(v)
    N = len(Pinv)
    if left:
        if nx != N - n:
            raise ValueError("when padding to the left, v=y and therefore nx must be = N-n")
    else:
        if nx != n:
            raise ValueError("when padding to the right, v=x and therefore nx must be = n")
    
    # z = padded v
    z = _padding(v,N,left,right)

    # A(z) = IDFT(1/P * DFT(z))
    z_fft = np.fft.fft(z)
    product = Pinv * z_fft
    result_complex = np.fft.ifft(product)
    imag_max = np.max(np.abs(result_complex.imag))
    real_max = np.max(np.abs(result_complex.real))

    assert imag_max / max(real_max, 1.0) < 1e-10, "result_complex has a large imaginary part"    
    result = result_complex.real    

    return result[:nx]

def _P_oof_func(freqs,sigma,fknee_hz,alpha,fmin_hz):
    '''
    Theoretical formula for 1/f power spectrum in frequency
    See "toast" engine in https://litebird-sim.readthedocs.io/en/latest/noise.html#litebird_sim.noise.add_one_over_f_noise
    '''
    f_alpha = np.abs(freqs)**alpha
    return sigma**2*(f_alpha + fknee_hz**alpha)/(f_alpha + fmin_hz**alpha)*len(freqs)

def inpainting_func(tod, nsamp_x, net_detector_ukrts, fknee_hz, alpha, fmin_hz, sampling_rate_hz, bin_size=1):
    '''
    Given a TOD chunk, [...]

    ARGUMENTS____________________________________________ 
    [...]
    '''    
    if fknee_hz < 0:
        raise ValueError("fknee_hz cannot be negative")
    if fmin_hz <= 0:
        raise ValueError("fmin_hz must be positive")
    if bin_size <= 0:
        raise ValueError("bin_size has to be a positive integer")

    nsamp_y = len(tod)

    if nsamp_y % bin_size != 0:
        raise ValueError("len(tod) must be divisible by bin_size")

    # reshape into a 2D grid and take the mean along the horizontal axis
    tod_binned = tod.reshape(-1, bin_size).mean(axis=1)
    nsamp_x_binned = nsamp_x//bin_size

    nsamp_tot_binned = (nsamp_x + nsamp_y)//bin_size
    sampling_rate_hz_binned = sampling_rate_hz/bin_size

    if nsamp_x % bin_size != 0:
        print("trucation because nsamp_x is not divisible by bin_size, shouldn't be a problem")

    # full frequencies and power spectrum
    freqs_binned = np.fft.fftfreq(nsamp_tot_binned, d=1/(sampling_rate_hz_binned))
    sigma_binned = net_detector_ukrts * np.sqrt(sampling_rate_hz_binned) / 1e6 #as in rescale_noise
    P_oof_binned = _P_oof_func(freqs_binned,sigma_binned,fknee_hz,alpha,fmin_hz)

    if np.any(P_oof_binned <= 0):
        raise ValueError("power spectrum must be positive")

    Pinv_oof_binned = 1/P_oof_binned

    b = -_A_func(Pinv_oof_binned, tod_binned, nsamp_x_binned, left=True)	#A_func applied to a vector [0,y]

    def _A_func_x_only(x):
        return _A_func(Pinv_oof_binned, x, nsamp_x_binned, right=True)	    #A_func applied to a vector [x,0]

    # define the LinearOperator for CG
    A_op = LinearOperator((nsamp_x_binned,nsamp_x_binned), matvec=_A_func_x_only)

    # initial guess for the CG solution, x0
    if fknee_hz > 0:
        nsamp_fknee = sampling_rate_hz/(fknee_hz)
        nsamp_avg = max(1, int(nsamp_fknee//2))
        
        avg_head = np.mean(tod[:nsamp_avg])
        avg_tail = np.mean(tod[-nsamp_avg:])
        x0 = avg_tail + np.arange(nsamp_x_binned)/nsamp_x_binned*(avg_head-avg_tail)
    else:
        x0 = np.zeros(nsamp_x_binned)

    x_sol, info = cg(A_op, b, rtol=1e-10, x0=x0)

    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")

    x = bin_size*(1/2 + np.arange(nsamp_x_binned))
    y = x_sol
    cs = CubicSpline(x, y)

    return cs(np.arange(nsamp_x))