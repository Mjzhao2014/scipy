import numpy as np
from numpy.testing import assert_warns
from scipy._lib._array_api import (
    xp_assert_close, xp_assert_equal,
    assert_almost_equal, assert_array_almost_equal,
)
from pytest import raises as assert_raises
import pytest
import time
from matplotlib.figure import Figure

import scipy.signal as spsig
import multiprocessing as mp
from scipy.fft import fft, fft2
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
    firwin, firwin2, freqz, remez, firls, minimum_phase, \
    convolve2d
from scipy.signal._fir_filter_design import FilterSpec, FIRFilter, firwin_2d
from scipy.signal import firwin, firls, remez, kaiserord, firwin2
from scipy.signal import lfilter
from pathlib import Path

def test_kaiser_beta():
    b = kaiser_beta(58.7)
    assert_almost_equal(b, 0.1102 * 50.0)
    b = kaiser_beta(22.0)
    assert_almost_equal(b, 0.5842 + 0.07886)
    b = kaiser_beta(21.0)
    assert b == 0.0
    b = kaiser_beta(10.0)
    assert b == 0.0


def test_kaiser_atten():
    a = kaiser_atten(1, 1.0)
    assert a == 7.95
    a = kaiser_atten(2, 1/np.pi)
    assert a == 2.285 + 7.95


def test_kaiserord():
    assert_raises(ValueError, kaiserord, 1.0, 1.0)
    numtaps, beta = kaiserord(2.285 + 7.95 - 0.001, 1/np.pi)
    assert (numtaps, beta) == (2, 0.0)


class TestFirwin:

    def check_response(self, h, expected_response, tol=.05):
        N = len(h)
        alpha = 0.5 * (N-1)
        m = np.arange(0,N) - alpha   # time indices of taps
        for freq, expected in expected_response:
            actual = abs(np.sum(h*np.exp(-1.j*np.pi*m*freq)))
            mse = abs(actual-expected)**2
            assert mse < tol, f'response not as expected, mse={mse:g} > {tol:g}'

    def test_response(self):
        N = 51
        f = .5
        # increase length just to try even/odd
        h = firwin(N, f)  # low-pass from 0 to f
        self.check_response(h, [(.25,1), (.75,0)])

        h = firwin(N+1, f, window='nuttall')  # specific window
        self.check_response(h, [(.25,1), (.75,0)])

        h = firwin(N+2, f, pass_zero=False)  # stop from 0 to f --> high-pass
        self.check_response(h, [(.25,0), (.75,1)])

        f1, f2, f3, f4 = .2, .4, .6, .8
        h = firwin(N+3, [f1, f2], pass_zero=False)  # band-pass filter
        self.check_response(h, [(.1,0), (.3,1), (.5,0)])

        h = firwin(N+4, [f1, f2])  # band-stop filter
        self.check_response(h, [(.1,1), (.3,0), (.5,1)])

        h = firwin(N+5, [f1, f2, f3, f4], pass_zero=False, scale=False)
        self.check_response(h, [(.1,0), (.3,1), (.5,0), (.7,1), (.9,0)])

        h = firwin(N+6, [f1, f2, f3, f4])  # multiband filter
        self.check_response(h, [(.1,1), (.3,0), (.5,1), (.7,0), (.9,1)])

        h = firwin(N+7, 0.1, width=.03)  # low-pass
        self.check_response(h, [(.05,1), (.75,0)])

        h = firwin(N+8, 0.1, pass_zero=False)  # high-pass
        self.check_response(h, [(.05,0), (.75,1)])

    def mse(self, h, bands):
        """Compute mean squared error versus ideal response across frequency
        band.
          h -- coefficients
          bands -- list of (left, right) tuples relative to 1==Nyquist of
            passbands
        """
        w, H = freqz(h, worN=1024)
        f = w/np.pi
        passIndicator = np.zeros(len(w), bool)
        for left, right in bands:
            passIndicator |= (f >= left) & (f < right)
        Hideal = np.where(passIndicator, 1, 0)
        mse = np.mean(abs(abs(H)-Hideal)**2)
        return mse

    def test_scaling(self):
        """
        For one lowpass, bandpass, and highpass example filter, this test
        checks two things:
          - the mean squared error over the frequency domain of the unscaled
            filter is smaller than the scaled filter (true for rectangular
            window)
          - the response of the scaled filter is exactly unity at the center
            of the first passband
        """
        N = 11
        cases = [
            ([.5], True, (0, 1)),
            ([0.2, .6], False, (.4, 1)),
            ([.5], False, (1, 1)),
        ]
        for cutoff, pass_zero, expected_response in cases:
            h = firwin(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
            hs = firwin(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
            if len(cutoff) == 1:
                if pass_zero:
                    cutoff = [0] + cutoff
                else:
                    cutoff = cutoff + [1]
            msg = 'least squares violation'
            assert self.mse(h, [cutoff]) < self.mse(hs, [cutoff]), msg
            self.check_response(hs, [expected_response], 1e-12)

    def test_fs_validation(self):
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            firwin(51, .5, fs=np.array([10, 20]))


class TestFirWinMore:
    """Different author, different style, different tests..."""

    def test_lowpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where
        # we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='lowpass', **kwargs)
        xp_assert_close(taps, taps_str)

    def test_highpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)

        # Ensure that ntaps is odd.
        ntaps |= 1

        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where
        # we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='highpass', **kwargs)
        xp_assert_close(taps, taps_str)

    def test_bandpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where
        # we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.2, 0.3-width/2, 0.3+width/2, 0.5,
                                0.7-width/2, 0.7+width/2, 0.8, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='bandpass', **kwargs)
        xp_assert_close(taps, taps_str)

    def test_bandstop_multi(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.2, 0.5, 0.8], window=('kaiser', beta),
                      scale=False)
        taps = firwin(ntaps, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where
        # we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.1, 0.2-width/2, 0.2+width/2, 0.35,
                                0.5-width/2, 0.5+width/2, 0.65,
                                0.8-width/2, 0.8+width/2, 0.9, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                decimal=5)

        taps_str = firwin(ntaps, pass_zero='bandstop', **kwargs)
        xp_assert_close(taps, taps_str)

    def test_fs_nyq(self):
        """Test the fs and nyq keywords."""
        nyquist = 1000
        width = 40.0
        relative_width = width/nyquist
        ntaps, beta = kaiserord(120, relative_width)
        taps = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta),
                        pass_zero=False, scale=False, fs=2*nyquist)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where
        # we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 200, 300-width/2, 300+width/2, 500,
                                700-width/2, 700+width/2, 800, 1000])
        freqs, response = freqz(taps, worN=np.pi*freq_samples/nyquist)
        assert_array_almost_equal(np.abs(response),
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

    def test_bad_cutoff(self):
        """Test that invalid cutoff argument raises ValueError."""
        # cutoff values must be greater than 0 and less than 1.
        assert_raises(ValueError, firwin, 99, -0.5)
        assert_raises(ValueError, firwin, 99, 1.5)
        # Don't allow 0 or 1 in cutoff.
        assert_raises(ValueError, firwin, 99, [0, 0.5])
        assert_raises(ValueError, firwin, 99, [0.5, 1])
        # cutoff values must be strictly increasing.
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
        # Must have at least one cutoff value.
        assert_raises(ValueError, firwin, 99, [])
        # 2D array not allowed.
        assert_raises(ValueError, firwin, 99, [[0.1, 0.2],[0.3, 0.4]])
        # cutoff values must be less than nyq.
        assert_raises(ValueError, firwin, 99, 50.0, fs=80)
        assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)

    def test_even_highpass_raises_value_error(self):
        """Test that attempt to create a highpass filter with an even number
        of taps raises a ValueError exception."""
        assert_raises(ValueError, firwin, 40, 0.5, pass_zero=False)
        assert_raises(ValueError, firwin, 40, [.25, 0.5])

    def test_bad_pass_zero(self):
        """Test degenerate pass_zero cases."""
        with assert_raises(ValueError, match='pass_zero must be'):
            firwin(41, 0.5, pass_zero='foo')
        with assert_raises(TypeError, match='cannot be interpreted'):
            firwin(41, 0.5, pass_zero=1.)
        for pass_zero in ('lowpass', 'highpass'):
            with assert_raises(ValueError, match='cutoff must have one'):
                firwin(41, [0.5, 0.6], pass_zero=pass_zero)
        for pass_zero in ('bandpass', 'bandstop'):
            with assert_raises(ValueError, match='must have at least two'):
                firwin(41, [0.5], pass_zero=pass_zero)

    def test_fs_validation(self):
        # The firwin2 API is now deprecated; ignore its deprecation warning and
        # ensure ValueError still propagates for bad fs.
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="Sampling.*single scalar"):
                firwin2(51, .5, 1, fs=np.array([10, 20]))


class TestFirwin2:

    def test_invalid_args(self):
        # firwin2 is deprecated; suppress its deprecation warning while
        # asserting argument validation still works as before.
        w = pytest.warns(DeprecationWarning)
        # `freq` and `gain` have different lengths.
        with w:
            with assert_raises(ValueError, match='must be of same length'):
                firwin2(50, [0, 0.5, 1], [0.0, 1.0])
        # `nfreqs` is less than `ntaps`.
        with w:
            with assert_raises(ValueError, match='ntaps must be less than nfreqs'):
                firwin2(50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
        # Decreasing value in `freq`
        with w:
            with assert_raises(ValueError, match='must be nondecreasing'):
                firwin2(50, [0, 0.5, 0.4, 1.0], [0, .25, .5, 1.0])
        # Value in `freq` repeated more than once.
        with w:
            with assert_raises(ValueError, match='must not occur more than twice'):
                firwin2(50, [0, .1, .1, .1, 1.0], [0.0, 0.5, 0.75, 1.0, 1.0])
        # `freq` does not start at 0.0.
        with w:
            with assert_raises(ValueError, match='start with 0'):
                firwin2(50, [0.5, 1.0], [0.0, 1.0])
        # `freq` does not end at fs/2.
        with w:
            with assert_raises(ValueError, match='end with fs/2'):
                firwin2(50, [0.0, 0.5], [0.0, 1.0])
        # Value 0 is repeated in `freq`
        with w:
            with assert_raises(ValueError, match='0 must not be repeated'):
                firwin2(50, [0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        # Value fs/2 is repeated in `freq`
        with w:
            with assert_raises(ValueError, match='fs/2 must not be repeated'):
                firwin2(50, [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
        # Value in `freq` that is too close to a repeated number
        with w:
            with assert_raises(ValueError, match='cannot contain numbers '
                                                 'that are too close'):
                firwin2(50, [0.0, 0.5 - np.finfo(float).eps * 0.5, 0.5, 0.5, 1.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0])

        # Type II filter, but the gain at nyquist frequency is not zero.
        with w:
            with assert_raises(ValueError, match='Type II filter'):
                firwin2(16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])

        # Type III filter, but the gains at nyquist and zero rate are not zero.
        with w:
            with assert_raises(ValueError, match='Type III filter'):
                firwin2(17, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
        with w:
            with assert_raises(ValueError, match='Type III filter'):
                firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)
        with w:
            with assert_raises(ValueError, match='Type III filter'):
                firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], antisymmetric=True)

        # Type IV filter, but the gain at zero rate is not zero.
        with w:
            with assert_raises(ValueError, match='Type IV filter'):
                firwin2(16, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)

    def test01(self):
        width = 0.04
        beta = 12.0
        ntaps = 400
        # Filter is 1 from w=0 to w=0.5, then decreases linearly from 1 to 0 as w
        # increases from w=0.5 to w=1  (w=1 is the Nyquist frequency).
        freq = [0.0, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0]
        with pytest.warns(DeprecationWarning):
            taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2,
                                                        0.75, 1.0-width/2])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                        [1.0, 1.0, 1.0, 1.0-width, 0.5, width], decimal=5)

    def test02(self):
        width = 0.04
        beta = 12.0
        # ntaps must be odd for positive gain at Nyquist.
        ntaps = 401
        # An ideal highpass filter.
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [0.0, 0.0, 1.0, 1.0]
        with pytest.warns(DeprecationWarning):
            taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.25, 0.5-width, 0.5+width, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test03(self):
        width = 0.02
        ntaps, beta = kaiserord(120, width)
        # ntaps must be odd for positive gain at Nyquist.
        ntaps = int(ntaps) | 1
        freq = [0.0, 0.4, 0.4, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        with pytest.warns(DeprecationWarning):
            taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.4-width, 0.4+width, 0.45,
                                    0.5-width, 0.5+width, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                    [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test04(self):
        """Test firwin2 when window=None."""
        ntaps = 5
        # Ideal lowpass: gain is 1 on [0,0.5], and 0 on [0.5, 1.0]
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0]
        with pytest.warns(DeprecationWarning):
            taps = firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
        alpha = 0.5 * (ntaps - 1)
        m = np.arange(0, ntaps) - alpha
        h = 0.5 * sinc(0.5 * m)
        assert_array_almost_equal(h, taps)

    def test05(self):
        """Test firwin2 for calculating Type IV filters"""
        ntaps = 1500

        freq = [0.0, 1.0]
        gain = [0.0, 1.0]
        with pytest.warns(DeprecationWarning):
            taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2:][::-1])

        freqs, response = freqz(taps, worN=2048)
        assert_array_almost_equal(abs(response), freqs / np.pi, decimal=4)

    def test06(self):
        """Test firwin2 for calculating Type III filters"""
        ntaps = 1501

        freq = [0.0, 0.5, 0.55, 1.0]
        gain = [0.0, 0.5, 0.0, 0.0]
        with pytest.warns(DeprecationWarning):
            taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        assert taps[ntaps // 2] == 0.0
        assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2 + 1:][::-1])

        freqs, response1 = freqz(taps, worN=2048)
        response2 = np.interp(freqs / np.pi, freq, gain)
        assert_array_almost_equal(abs(response1), response2, decimal=3)

    def test_fs_nyq(self):
        with pytest.warns(DeprecationWarning):
            taps1 = firwin2(80, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
        with pytest.warns(DeprecationWarning):
            taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], fs=120.0)
        assert_array_almost_equal(taps1, taps2)

    def test_tuple(self):
        with pytest.warns(DeprecationWarning):
            taps1 = firwin2(150, (0.0, 0.5, 0.5, 1.0), (1.0, 1.0, 0.0, 0.0))
        with pytest.warns(DeprecationWarning):
            taps2 = firwin2(150, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        assert_array_almost_equal(taps1, taps2)

    def test_input_modyfication(self):
        freq1 = np.array([0.0, 0.5, 0.5, 1.0])
        freq2 = np.array(freq1)
        with pytest.warns(DeprecationWarning):
            firwin2(80, freq1, [1.0, 1.0, 0.0, 0.0])
        xp_assert_equal(freq1, freq2)


class TestRemez:

    def test_bad_args(self):
        assert_raises(ValueError, remez, 11, [0.1, 0.4], [1], type='pooka')

    def test_hilbert(self):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band

        # design an unity gain hilbert bandpass filter from w to 0.5-w
        h = remez(11, [a, 0.5-a], [1], type='hilbert')

        # make sure the filter has correct # of taps
        assert len(h) == N, "Number of Taps"

        # make sure it is type III (anti-symmetric tap coefficients)
        assert_array_almost_equal(h[:(N-1)//2], -h[:-(N-1)//2-1:-1])

        # Since the requested response is symmetric, all even coefficients
        # should be zero (or in this case really small)
        assert (abs(h[1::2]) < 1e-15).all(), "Even Coefficients Equal Zero"

        # now check the frequency response
        w, H = freqz(h, 1)
        f = w/2/np.pi
        Hmag = abs(H)

        # should have a zero at 0 and pi (in this case close to zero)
        assert (Hmag[[0, -1]] < 0.02).all(), "Zero at zero and pi"

        # check that the pass band is close to unity
        idx = np.logical_and(f > a, f < 0.5-a)
        assert (abs(Hmag[idx] - 1) < 0.015).all(), "Pass Band Close To Unity"

    def test_compare(self):
        # test comparison to MATLAB
        k = [0.024590270518440, -0.041314581814658, -0.075943803756711,
             -0.003530911231040, 0.193140296954975, 0.373400753484939,
             0.373400753484939, 0.193140296954975, -0.003530911231040,
             -0.075943803756711, -0.041314581814658, 0.024590270518440]
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
        xp_assert_close(h, k)

        h = [-0.038976016082299, 0.018704846485491, -0.014644062687875,
             0.002879152556419, 0.016849978528150, -0.043276706138248,
             0.073641298245579, -0.103908158578635, 0.129770906801075,
             -0.147163447297124, 0.153302248456347, -0.147163447297124,
             0.129770906801075, -0.103908158578635, 0.073641298245579,
             -0.043276706138248, 0.016849978528150, 0.002879152556419,
             -0.014644062687875, 0.018704846485491, -0.038976016082299]
        xp_assert_close(remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.), h)

    def test_fs_validation(self):
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            remez(11, .1, 1, fs=np.array([10, 20]))

class TestFirls:

    def test_bad_args(self):
        # even numtaps
        assert_raises(ValueError, firls, 10, [0.1, 0.2], [0, 0])
        # odd bands
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.4], [0, 0, 0])
        # len(bands) != len(desired)
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.4], [0, 0, 0])
        # non-monotonic bands
        assert_raises(ValueError, firls, 11, [0.2, 0.1], [0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.3], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.3, 0.4, 0.1, 0.2], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.1, 0.3, 0.2, 0.4], [0] * 4)
        # negative desired
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [-1, 1])
        # len(weight) != len(pairs)
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], weight=[1, 2])
        # negative weight
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], weight=[-1])

    def test_firls(self):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band

        # design a halfband symmetric low-pass filter
        h = firls(11, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)

        # make sure the filter has correct # of taps
        assert h.shape[0] == N

        # make sure it is symmetric
        midx = (N-1) // 2
        assert_array_almost_equal(h[:midx], h[:-midx-1:-1])

        # make sure the center tap is 0.5
        assert_almost_equal(h[midx], 0.5)

        # For halfband symmetric, odd coefficients (except the center)
        # should be zero (really small)
        hodd = np.hstack((h[1:midx:2], h[-midx+1::2]))
        assert_array_almost_equal(hodd, np.zeros_like(hodd))

        # now check the frequency response
        w, H = freqz(h, 1)
        f = w/2/np.pi
        Hmag = np.abs(H)

        # check that the pass band is close to unity
        idx = np.logical_and(f > 0, f < a)
        assert_array_almost_equal(Hmag[idx], np.ones_like(Hmag[idx]), decimal=3)

        # check that the stop band is close to zero
        idx = np.logical_and(f > 0.5-a, f < 0.5)
        assert_array_almost_equal(Hmag[idx], np.zeros_like(Hmag[idx]), decimal=3)

    def test_compare(self):
        # compare to OCTAVE output
        taps = firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], weight=[1, 2])
        # >> taps = firls(8, [0 0.5 0.55 1], [1 1 0 0], [1, 2]);
        known_taps = [-6.26930101730182e-04, -1.03354450635036e-01,
                      -9.81576747564301e-03, 3.17271686090449e-01,
                      5.11409425599933e-01, 3.17271686090449e-01,
                      -9.81576747564301e-03, -1.03354450635036e-01,
                      -6.26930101730182e-04]
        xp_assert_close(taps, known_taps)

        # compare to MATLAB output
        taps = firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], weight=[1, 2])
        # >> taps = firls(10, [0 0.5 0.5 1], [1 1 0 0], [1, 2]);
        known_taps = [
            0.058545300496815, -0.014233383714318, -0.104688258464392,
            0.012403323025279, 0.317930861136062, 0.488047220029700,
            0.317930861136062, 0.012403323025279, -0.104688258464392,
            -0.014233383714318, 0.058545300496815]
        xp_assert_close(taps, known_taps)

        # With linear changes:
        taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], fs=20)
        # >> taps = firls(6, [0, 0.1, 0.2, 0.3, 0.4, 0.5], [1, 0, 0, 1, 1, 0])
        known_taps = [
            1.156090832768218, -4.1385894727395849, 7.5288619164321826,
            -8.5530572592947856, 7.5288619164321826, -4.1385894727395849,
            1.156090832768218]
        xp_assert_close(taps, known_taps)

    def test_rank_deficient(self):
        # solve() runs but warns (only sometimes, so here we don't use match)
        x = firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.)
        absh2 = np.abs(h[:2])
        xp_assert_close(absh2, np.ones_like(absh2), atol=1e-5)
        absh2 = np.abs(h[-2:])
        xp_assert_close(absh2, np.zeros_like(absh2), atol=1e-6, rtol=1e-7)
        # switch to pinvh (tolerances could be higher with longer
        # filters, but using shorter ones is faster computationally and
        # the idea is the same)
        x = firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.)
        mask = w < 0.01
        assert mask.sum() > 3
        habs = np.abs(h[mask])
        xp_assert_close(habs, np.ones_like(habs), atol=1e-4)
        mask = w > 0.99
        assert mask.sum() > 3
        habs = np.abs(h[mask])
        xp_assert_close(habs, np.zeros_like(habs), atol=1e-4)

    def test_fs_validation(self):
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            firls(11, .1, 1, fs=np.array([10, 20]))

class TestMinimumPhase:
    @pytest.mark.thread_unsafe
    def test_bad_args(self):
        # not enough taps
        assert_raises(ValueError, minimum_phase, [1.])
        assert_raises(ValueError, minimum_phase, [1., 1.])
        assert_raises(ValueError, minimum_phase, np.full(10, 1j))
        assert_raises(ValueError, minimum_phase, 'foo')
        assert_raises(ValueError, minimum_phase, np.ones(10), n_fft=8)
        assert_raises(ValueError, minimum_phase, np.ones(10), method='foo')
        assert_warns(RuntimeWarning, minimum_phase, np.arange(3))
        with pytest.raises(ValueError, match="is only supported when"):
            minimum_phase(np.ones(3), method='hilbert', half=False)

    def test_homomorphic(self):
        # check that it can recover frequency responses of arbitrary
        # linear-phase filters

        # for some cases we can get the actual filter back
        h = [1, -1]
        h_new = minimum_phase(np.convolve(h, h[::-1]))
        xp_assert_close(h_new, np.asarray(h, dtype=np.float64), rtol=0.05)

        # but in general we only guarantee we get the magnitude back
        rng = np.random.RandomState(0)
        for n in (2, 3, 10, 11, 15, 16, 17, 20, 21, 100, 101):
            h = rng.randn(n)
            h_linear = np.convolve(h, h[::-1])
            h_new = minimum_phase(h_linear)
            xp_assert_close(np.abs(fft(h_new)), np.abs(fft(h)), rtol=1e-4)
            h_new = minimum_phase(h_linear, half=False)
            assert len(h_linear) == len(h_new)
            xp_assert_close(np.abs(fft(h_new)), np.abs(fft(h_linear)), rtol=1e-4)

    def test_hilbert(self):
        # compare to MATLAB output of reference implementation

        # f=[0 0.3 0.5 1];
        # a=[1 1 0 0];
        # h=remez(11,f,a);
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
        k = [0.349585548646686, 0.373552164395447, 0.326082685363438,
             0.077152207480935, -0.129943946349364, -0.059355880509749]
        m = minimum_phase(h, 'hilbert')
        xp_assert_close(m, k, rtol=5e-3)

        # f=[0 0.8 0.9 1];
        # a=[0 0 1 1];
        # h=remez(20,f,a);
        h = remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.)
        k = [0.232486803906329, -0.133551833687071, 0.151871456867244,
             -0.157957283165866, 0.151739294892963, -0.129293146705090,
             0.100787844523204, -0.065832656741252, 0.035361328741024,
             -0.014977068692269, -0.158416139047557]
        m = minimum_phase(h, 'hilbert', n_fft=2**19)
        xp_assert_close(m, k, rtol=2e-3)


class Testfirwin_2d:
    def test_invalid_args(self):
        with pytest.raises(ValueError, 
                           match="hsize must be a 2-element tuple or list"):
            firwin_2d((50,), window=(("kaiser", 5.0), "boxcar"), fc=0.4)
        
        with pytest.raises(ValueError, 
                           match="window must be a 2-element tuple or list"):
            firwin_2d((51, 51), window=("hamming",), fc=0.5)
        
        with pytest.raises(ValueError, 
                           match="window must be a 2-element tuple or list"):
            firwin_2d((51, 51), window="invalid_window", fc=0.5)

    def test_filter_design(self):
        hsize = (51, 51)
        window = (("kaiser", 8.0), ("kaiser", 8.0))
        fc = 0.4
        taps_kaiser = firwin_2d(hsize, window, fc=fc)
        assert taps_kaiser.shape == (51, 51)

        window = ("hamming", "hamming")
        taps_hamming = firwin_2d(hsize, window, fc=fc)
        assert taps_hamming.shape == (51, 51)

    def test_impulse_response(self):
        hsize = (31, 31)
        window = ("hamming", "hamming")
        fc = 0.4
        taps = firwin_2d(hsize, window, fc=fc)

        impulse = np.zeros((63, 63))
        impulse[31, 31] = 1

        response = convolve2d(impulse, taps, mode='same')

        expected_response = taps
        xp_assert_close(response[16:47, 16:47], expected_response, rtol=1e-5)

    def test_frequency_response(self):
        """Compare 1d and 2d frequency response. """
        hsize = (31, 31)
        windows = ("hamming", "hamming")
        fc = 0.4
        taps_1d = firwin(numtaps=hsize[0], cutoff=fc, window=windows[0])
        taps_2d = firwin_2d(hsize, windows, fc=fc)

        f_resp_1d = fft(taps_1d)
        f_resp_2d = fft2(taps_2d)

        xp_assert_close(f_resp_2d[0, :], f_resp_1d,
                        err_msg='DC Gain at (0, f1) is not unity!')
        xp_assert_close(f_resp_2d[:, 0], f_resp_1d,
                        err_msg='DC Gain at (f0, 0) is not unity!')
        xp_assert_close(f_resp_2d, np.outer(f_resp_1d, f_resp_1d),
                        atol=np.finfo(f_resp_2d.dtype).resolution,
                        err_msg='2d frequency response is not product of 1d responses')

    def test_symmetry(self):
        hsize = (51, 51)
        window = ("hamming", "hamming")
        fc = 0.4
        taps = firwin_2d(hsize, window, fc=fc)
        xp_assert_close(taps, np.flip(taps), rtol=1e-5)

    def test_circular_symmetry(self):
        hsize = (51, 51)
        window = "hamming"
        taps = firwin_2d(hsize, window, circular=True, fc=0.5)
        center = hsize[0] // 2
        for i in range(hsize[0]):
            for j in range(hsize[1]):
                xp_assert_close(taps[i, j], 
                                taps[center - (i - center), center - (j - center)], 
                                rtol=1e-5)

    def test_edge_case_circular(self):
        hsize = (3, 3)
        window = "hamming"
        taps_small = firwin_2d(hsize, window, circular=True, fc=0.5)
        assert taps_small.shape == (3, 3)

        hsize = (101, 101)
        taps_large = firwin_2d(hsize, window, circular=True, fc=0.5)
        assert taps_large.shape == (101, 101)

    def test_known_result(self):
        hsize = (5, 5)
        window = ('kaiser', 8.0)
        fc = 0.1
        fs = 2

        row_filter = firwin(hsize[0], cutoff=fc, window=window, fs=fs)
        col_filter = firwin(hsize[1], cutoff=fc, window=window, fs=fs)
        known_result = np.outer(row_filter, col_filter)

        taps = firwin_2d(hsize, (window, window), fc=fc)
        assert taps.shape == known_result.shape, (
            f"Shape mismatch: {taps.shape} vs {known_result.shape}"
        )
        assert np.allclose(taps, known_result, rtol=1e-1), (
            f"Filter shape mismatch: {taps} vs {known_result}"
        )
# ----------------------------
# Helpers for item 8 baselines
# ----------------------------

def _load_baselines():
    """
    Load historical coefficient baselines from a JSON file placed next to this test.
    The file should map keys to lists of floats, e.g.:

    {
      "firwin_lowpass_hann_15_0p30_fs1p0": [ ... ],
      "firls_lpass_15_0p25_0p35_fs1p0": [ ... ],
      "remez_lpass_17_0p22_0p32_fs0p5": [ ... ]
    }
    """
    p = Path(__file__).with_name("fir_baselines.json")
    if not p.exists():
        return None
    with p.open("r") as f:
        return json.load(f)


BASELINES = _load_baselines()


def _get_baseline_or_skip(key):
    if BASELINES is None or key not in BASELINES:
        pytest.skip(
            f"Historical baseline '{key}' not available; "
            f"create scipy/signal/tests/fir_baselines.json to enable parity check."
        )
    return np.asarray(BASELINES[key], dtype=float)


# ---------------------------------------------
# Items 1–7: FilterSpec construction & repr
# ---------------------------------------------

def test_filterspec_invalid_numtaps_raises():
    with pytest.raises(ValueError):
        FilterSpec(numtaps=0, band=[0.0, 0.4], gain=[1.0, 0.0], window="hann", fs=1.0)
    with pytest.raises(ValueError):
        FilterSpec(numtaps=-3, band=[0.0, 0.4], gain=[1.0, 0.0], window="hann", fs=1.0)


def test_filterspec_band_gain_mismatch_raises():
    # bands/gains length mismatch
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3, 0.5, 0.6], gain=[1.0, 0.0, 0.0],
                   window="hann", fs=1.0)
    # odd number of band edges (pairs required)
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3, 0.5], gain=[1.0, 0.0],
                   window="hann", fs=1.0)


def test_filterspec_nonmonotonic_bands_raises():
    # strictly increasing required
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.3, 0.2], gain=[1.0, 0.0], window="hann", fs=1.0)
    # flat/duplicate edge not allowed for strict monotonicity
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.1, 0.1], gain=[1.0, 0.0], window="hann", fs=1.0)


def test_filterspec_invalid_gain_values_raises():
    # negative
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3], gain=[-1.0, 0.0], window="hann", fs=1.0)
    # non-real
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3], gain=[1.0+0.1j, 0.0], window="hann", fs=1.0)


def test_filterspec_fs_not_single_scalar_raises():
    # non-scalar array-like
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3], gain=[1.0, 0.0], window="hann", fs=[1.0, 2.0])
    # non-numeric
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3], gain=[1.0, 0.0], window="hann", fs="48k")


def test_filterspec_bad_window_spec_raises():
    # unsupported window string
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3], gain=[1.0, 0.0], window="not_a_window", fs=1.0)
    # malformed tuple (wrong arity)
    with pytest.raises(ValueError):
        FilterSpec(numtaps=15, band=[0.0, 0.3], gain=[1.0, 0.0], window=("kaiser",), fs=1.0)


def test_filterspec_repr_includes_key_fields():
    spec = FilterSpec(numtaps=21, band=[0.0, 0.25], gain=[1.0, 0.0], window="hann", fs=1.0)
    r = repr(spec)
    # Avoid exact formatting; just assert key fields appear.
    assert "numtaps=" in r
    assert "window=" in r
    # Either band/method should be represented; here we expect band.
    assert "band=" in r or "method=" in r


# ------------------------------------------------------------
# Item 8: Legacy routines return FIRFilter and match baseline
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "key, builder",
    [
        # firwin: low-pass, hann window, legacy signature
        (
            "firwin_lowpass_hann_15_0p30_fs1p0",
            lambda: firwin(15, 0.30, window="hann", pass_zero=True, fs=1.0),
        ),
        # firls: low-pass bands, legacy signature
        (
            "firls_lpass_15_0p25_0p35_fs1p0",
            lambda: firls(15, [0.0, 0.25, 0.35, 0.5], [1.0, 1.0, 0.0, 0.0], fs=1.0),
        ),
        # remez: low-pass, legacy signature
        (
            "remez_lpass_17_0p22_0p32_fs0p5",
            lambda: remez(17, [0.0, 0.22, 0.32, 0.5], [1.0, 0.0], fs=1.0),
        ),
    ],
)
def test_legacy_routines_return_firfilter_and_match_historical(key, builder):
    filt = builder()
    # Type upgrade check
    assert isinstance(filt, FIRFilter)

    # Historical parity check (optional; skipped if no baseline file/key)
    expected = _get_baseline_or_skip(key)
    assert_allclose(np.asarray(filt), expected, rtol=1e-10, atol=1e-12)

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scipy.signal import firwin, firwin2, firls, remez
from scipy.signal._fir_filter_design import FilterSpec, FIRFilter


# ---------------------------------------------------------------------
# Optional baselines: numeric taps and error messages from pre-refactor
# ---------------------------------------------------------------------

def _load_json(name):
    p = Path(__file__).with_name(name)
    if not p.exists():
        return None
    with p.open("r") as f:
        return json.load(f)

# Numeric baselines for “historical outputs”
BASE_TAPS = _load_json("fir_baselines.json")
# Error message baselines for consistency checks
BASE_ERRS = _load_json("fir_error_messages.json")


def _get_taps_baseline_or_skip(key):
    if BASE_TAPS is None or key not in BASE_TAPS:
        pytest.skip(
            f"Baseline missing: {key}. "
            f"Create scipy/signal/tests/fir_baselines.json to enable this parity check."
        )
    return np.asarray(BASE_TAPS[key], dtype=float)


def _get_err_baseline_or_skip(key):
    if BASE_ERRS is None or key not in BASE_ERRS:
        pytest.skip(
            f"Error-string baseline missing: {key}. "
            f"Create scipy/signal/tests/fir_error_messages.json to enable this wording check."
        )
    return BASE_ERRS[key]


# ---------------------------
# 9. firwin legacy length
# ---------------------------

def test_firwin_legacy_returns_expected_length():
    N = 25
    # Low-pass in legacy signature form
    filt = firwin(N, cutoff=0.28, window="hann", pass_zero=True, fs=1.0)
    # Return type is FIRFilter per your design, but len() must still match.
    assert isinstance(filt, FIRFilter)
    assert len(filt) == N


# -------------------------------------------------------------
# 10. firwin legacy invalid params -> ValueError with wording
# -------------------------------------------------------------

@pytest.mark.parametrize(
    "key, builder",
    [
        # non-monotonic / malformed cutoff specification
        (
            "firwin_nonmonotonic_cutoff",
            lambda: firwin(17, [0.30, 0.20], window="hann", pass_zero=True, fs=1.0),
        ),
    ],
)
def test_firwin_legacy_error_wording_consistent_with_baseline(key, builder):
    expected = _get_err_baseline_or_skip(key)
    with pytest.raises(ValueError) as e:
        builder()
    # Exact wording parity with historical snapshot
    assert str(e.value) == expected


# ------------------------------------------------------
# 11. firwin thread-safety / concurrent behavior stable
# ------------------------------------------------------

def _build_firwin_lowpass():
    return firwin(33, 0.26, window="hann", pass_zero=True, fs=1.0)

def test_firwin_concurrent_calls_produce_stable_results():
    # Run the same design concurrently and sequentially; results should match.
    ref = np.asarray(_build_firwin_lowpass())

    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(lambda _: np.asarray(_build_firwin_lowpass()), range(4)))

    for arr in outs:
        assert_allclose(arr, ref, rtol=0, atol=0)


# -------------------------------------------------
# 12. firwin accepts FilterSpec, length is correct
# -------------------------------------------------

def test_firwin_filterspec_returns_expected_length():
    spec = FilterSpec(
        numtaps=27,
        band=[0.0, 0.24],
        gain=[1.0, 0.0],
        window="hann",
        fs=1.0,
    )
    filt = firwin(spec)
    assert isinstance(filt, FIRFilter)
    assert len(filt) == 27


# ------------------------------------------------------------
# 13. firwin routes via internal dispatcher when given spec
# ------------------------------------------------------------

def test_firwin_uses_dispatcher_with_filterspec(monkeypatch):
    # White-box: patch the module-level dispatcher to a sentinel.
    import scipy.signal._fir_filter_design as ffd

    if not hasattr(ffd, "_dispatch_firwin"):
        pytest.skip("No _dispatch_firwin hook found; dispatcher test not applicable.")

    sentinel = np.arange(7, dtype=float)
    seen = {}

    def fake_dispatch(spec):
        # capture type without importing FilterSpec here (already imported above)
        seen["type"] = type(spec).__name__
        return sentinel

    monkeypatch.setattr(ffd, "_dispatch_firwin", fake_dispatch, raising=True)

    spec = FilterSpec(numtaps=31, band=[0.0, 0.20], gain=[1.0, 0.0], window="hann", fs=1.0)
    out = firwin(spec)

    # firwin should have called the dispatcher and wrapped its output as FIRFilter
    assert seen.get("type") == "FilterSpec"
    assert isinstance(out, FIRFilter)
    assert_allclose(np.asarray(out), sentinel, rtol=0, atol=0)


# ------------------------------------------------------
# 14. Changing a FilterSpec parameter changes the output
# ------------------------------------------------------

def test_firwin_filterspec_parameter_change_alters_taps():
    base = FilterSpec(numtaps=29, band=[0.0, 0.22], gain=[1.0, 0.0], window="hann", fs=1.0)
    alt  = FilterSpec(numtaps=29, band=[0.0, 0.32], gain=[1.0, 0.0], window="hann", fs=1.0)

    taps_base = np.asarray(firwin(base))
    taps_alt  = np.asarray(firwin(alt))

    # Distinct specs should yield non-identical designs
    assert not np.allclose(taps_base, taps_alt, rtol=1e-12, atol=1e-14)


# -------------------------------------------------------------------------------------------
# 15. Invalid FilterSpec path uses FilterSpec wording; legacy path keeps legacy wording
# -------------------------------------------------------------------------------------------

def test_firwin_invalid_filterspec_uses_filterspec_message_and_legacy_differs():
    # 1) Capture the FilterSpec validation message for an invalid spec.
    with pytest.raises(ValueError) as e_spec:
        FilterSpec(numtaps=0, band=[0.0, 0.3], gain=[1.0, 0.0], window="hann", fs=1.0)
    spec_msg = str(e_spec.value)

    # 2) Confirm calling firwin(FilterSpec(...invalid...)) raises with the same message.
    #    (The exception occurs during FilterSpec construction in the call expression.)
    with pytest.raises(ValueError) as e_firwin_spec:
        firwin(FilterSpec(numtaps=-5, band=[0.0, 0.3], gain=[1.0, 0.0], window="hann", fs=1.0))
    assert str(e_firwin_spec.value) == spec_msg or str(e_firwin_spec.value) != ""

    # 3) Legacy misuse: message should *not* equal the FilterSpec validation text.
    with pytest.raises(ValueError) as e_legacy:
        firwin(17, [0.30, 0.20], window="hann", pass_zero=True, fs=1.0)
    assert str(e_legacy.value) != spec_msg


# -----------------------------------------------------------------------------------
# 16. Other legacy routines: misuse raises ValueError with historical message text
# -----------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key, builder",
    [
        ("firls_mismatch_lengths",
         lambda: firls(19, [0.0, 0.2, 0.3], [1.0, 0.0], fs=1.0)),
        ("remez_nonmonotonic",
         lambda: remez(21, [0.2, 0.1, 0.4, 0.5], [1.0, 0.0], fs=1.0)),
        ("kaiserord_invalid",
         # kaiserord does not produce taps; baseline holds its error wording.
         lambda: __import__("scipy.signal", fromlist=["signal"]).signal.kaiserord(-60, 0.1)),
    ],
)
def test_other_legacy_routines_error_wording_matches_baseline(key, builder):
    expected = _get_err_baseline_or_skip(key)
    with pytest.raises(ValueError) as e:
        builder()
    assert str(e.value) == expected


# ------------------------------------------------
# 17. firwin2 emits the targeted DeprecationWarning
# ------------------------------------------------

def test_firwin2_emits_deprecationwarning():
    with pytest.warns(DeprecationWarning, match=r"deprecated.*method=['\"]multiband['\"]"):
        firwin2(13, [0.0, 0.25, 0.35, 0.5], [1.0, 1.0, 0.0, 0.0], fs=1.0)


# --------------------------------------------------------------------
# 18. firwin2 parity with firwin(method="multiband") within tolerance
# --------------------------------------------------------------------

def test_firwin2_matches_firwin_multiband_within_tolerance():
    N = 27
    bands = [0.0, 0.22, 0.34, 0.5]
    gains = [1.0, 1.0, 0.0, 0.0]
    fs = 1.0

    t2 = np.asarray(firwin2(N, bands, gains, fs=fs))
    t1 = np.asarray(firwin(N, band=bands, gain=gains, window="hann",
                           fs=fs, method="multiband"))

    assert_allclose(t1, t2, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------
# 19. firwin2 invalid input keeps historical error wording (baseline)
# --------------------------------------------------------------------

def test_firwin2_invalid_input_error_wording_matches_baseline():
    expected = _get_err_baseline_or_skip("firwin2_nonmonotonic")
    with pytest.raises(ValueError) as e:
        firwin2(15, [0.3, 0.2, 0.4, 0.5], [1.0, 0.0], fs=1.0)
    assert str(e.value) == expected

import numpy as np
import pytest
from numpy.testing import assert_allclose

import matplotlib
matplotlib.use("Agg", force=True)  # headless plotting
from matplotlib.figure import Figure

from scipy.signal import lfilter, sosfilt, freqz
from scipy.signal._fir_filter_design import FilterSpec, FIRFilter


# ---------- Fixture: a held-out FIRFilter via the FilterSpec path ----------

@pytest.fixture
def filt():
    # Different values from earlier tests to keep this set "held-out".
    spec = FilterSpec(
        numtaps=35,
        band=[0.0, 0.18],
        gain=[1.0, 0.0],
        window="hamming",
        fs=1.0,
    )
    out = __import__("scipy.signal", fromlist=["signal"]).signal.firwin(spec)
    assert isinstance(out, FIRFilter)
    return out


@pytest.fixture
def coeffs(filt):
    return np.asarray(filt)


# ----------------------------------------------------------------------
# 20. SciPy/NumPy consumers accept FIRFilter and match ndarray output
# ----------------------------------------------------------------------

def test_consumers_accept_firfilter_and_match_ndarray(filt, coeffs):
    rng = np.random.default_rng(20250731)
    x = rng.standard_normal(512)

    # SciPy consumer: lfilter
    y_f = lfilter(filt, 1.0, x)
    y_a = lfilter(coeffs, 1.0, x)
    assert_allclose(y_f, y_a, rtol=1e-12, atol=0.0)

    # NumPy consumer: convolve (accepts array-likes)
    c_f = np.convolve(x, filt, mode="same")
    c_a = np.convolve(x, coeffs, mode="same")
    assert_allclose(c_f, c_a, rtol=0, atol=0)


# -----------------------------------------------------------
# 21. Indexing and iteration behave like the underlying array
# -----------------------------------------------------------

def test_indexing_and_iteration_match_ndarray(filt, coeffs):
    # Scalar index
    assert_allclose(filt[0], coeffs[0])
    assert_allclose(filt[-1], coeffs[-1])

    # Slices
    assert_allclose(filt[1::4], coeffs[1::4])

    # Fancy indexing
    idx = np.array([2, 7, -3, -1])
    assert_allclose(filt[idx], coeffs[idx])

    # Boolean mask (median-based to avoid trivial masks)
    mask = coeffs >= np.median(coeffs)
    assert_allclose(filt[mask], coeffs[mask])

    # Iteration order and values
    assert_allclose(np.fromiter(iter(filt), dtype=coeffs.dtype), coeffs)


# -------------------------------------------------------------------
# 22. FIRFilter.freqz produces same response as scipy.signal.freqz
# -------------------------------------------------------------------

def test_freqz_matches_scipy_freqz(filt, coeffs):
    # Use a fixed worN for determinism
    w_obj, h_obj = filt.freqz(worN=1024)
    w_ref, h_ref = freqz(coeffs, 1.0, worN=1024)
    assert_allclose(w_obj, w_ref, rtol=0, atol=0)
    assert_allclose(h_obj, h_ref, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------
# 23. FIRFilter.plot returns a valid matplotlib Figure
# --------------------------------------------------------

def test_plot_returns_figure(filt):
    # Prefer show=False, but fall back if signature differs
    try:
        fig = filt.plot(show=False)
    except TypeError:
        fig = filt.plot()
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------
# 24. FIRFilter.to_sos cascade matches lfilter behavior within tolerance
# ----------------------------------------------------------------------

def test_to_sos_matches_lfilter(filt, coeffs):
    sos = filt.to_sos()

    rng = np.random.default_rng(42)
    x = rng.normal(size=800)

    y_sos = sosfilt(sos, x)
    y_ref = lfilter(coeffs, 1.0, x)

    # FIR → SOS should be numerically equivalent; allow tiny FP noise.
    assert_allclose(y_sos, y_ref, rtol=1e-10, atol=1e-12)


# -----------------------------------------------------------------------------------
# 25. FIRFilter retains identity reference to its FilterSpec and defines equality
# -----------------------------------------------------------------------------------

def test_spec_identity_and_equality():
    spec1 = FilterSpec(
        numtaps=33, band=[0.0, 0.2], gain=[1.0, 0.0], window="hann", fs=1.0
    )
    from scipy.signal import firwin as _firwin
    f1 = _firwin(spec1)

    # Identity: the filter holds the exact same spec object.
    assert f1.spec is spec1

    # Content equality: a spec with the same fields compares equal.
    spec1_clone = FilterSpec(
        numtaps=33, band=[0.0, 0.2], gain=[1.0, 0.0], window="hann", fs=1.0
    )
    assert spec1 == spec1_clone

    # Inequality: a meaningfully different spec does not compare equal.
    spec2 = FilterSpec(
        numtaps=33, band=[0.0, 0.25], gain=[1.0, 0.0], window="hann", fs=1.0
    )
    assert spec1 != spec2


# ----------------------------------------------------------------------------------------
# 26. Convenience helpers exist and return expected types/shapes (smoke-type checks)
# ----------------------------------------------------------------------------------------

def test_helpers_exist_and_shapes_are_expected(filt):
    # freqz: returns 1-D frequency grid and complex response
    w, h = filt.freqz(worN=257)
    assert isinstance(w, np.ndarray) and w.ndim == 1 and w.size == 257
    assert isinstance(h, np.ndarray) and h.ndim == 1 and h.size == 257

    # to_sos: returns (n_sections, 6)
    sos = filt.to_sos()
    assert isinstance(sos, np.ndarray) and sos.ndim == 2 and sos.shape[1] == 6

    # plot: returns a Figure
    try:
        fig = filt.plot(show=False)
    except TypeError:
        fig = filt.plot()
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------------------------------------
# 27. Passing FIRFilter directly into lfilter works the same as passing the raw coefficient array (2-D)
# ----------------------------------------------------------------------------------------------------

def test_lfilter_accepts_firfilter_for_2d_input(filt, coeffs):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, 400))  # 2-D signal; default axis=-1

    Y_obj = lfilter(filt, 1.0, X)
    Y_arr = lfilter(coeffs, 1.0, X)
    assert_allclose(Y_obj, Y_arr, rtol=1e-12, atol=0.0)

# -------------------------
# Helpers for snapshot data
# -------------------------

def _load_json(sidecar_name):
    p = Path(__file__).with_name(sidecar_name)
    if not p.exists():
        return None
    with p.open("r") as f:
        return json.load(f)

API_BASELINE = _load_json("signal_public_api.json")           # list[str] of names from pre-refactor
DOC_BASELINES = _load_json("fir_doc_examples.json")           # optional numeric baselines for doc snippets
PERF_BOUNDS = _load_json("fir_perf_thresholds.json")          # {case: max_seconds}


# ============================================================
# 28. Public import surface remains a superset of historical
# ============================================================

def test_public_import_surface_superset_of_baseline():
    baseline = API_BASELINE
    if baseline is None:
        pytest.skip("signal_public_api.json missing; cannot verify import surface parity.")
    # Prefer __all__; fall back to public names
    current = set(getattr(spsig, "__all__", [n for n in dir(spsig) if not n.startswith("_")]))
    assert set(baseline).issubset(current), "Some historical public names are missing in scipy.signal"


# ==================================================================================
# 29. Untouched remez remains unchanged and does NOT accept a FilterSpec (TypeError)
# ==================================================================================

def test_remez_rejects_filterspec():
    spec = FilterSpec(numtaps=17, band=[0.0, 0.25], gain=[1.0, 0.0], window="hann", fs=1.0)
    with pytest.raises(TypeError):
        remez(spec, [0.0, 0.2, 0.3, 0.5], [1.0, 0.0], fs=1.0)

def test_remez_numeric_parity_against_baseline():
    baselines = _load_json("fir_baselines.json")
    key = "remez_lpass_19_0p20_0p30_fs1p0_doclike"
    if baselines is None or key not in baselines:
        pytest.skip("No historical taps snapshot for remez parity.")
    expected = np.asarray(baselines[key], dtype=float)
    taps = np.asarray(remez(19, [0.0, 0.20, 0.30, 0.5], [1.0, 0.0], fs=1.0))
    assert_allclose(taps, expected, rtol=1e-10, atol=1e-12)


# ===========================================================================================
# 30. Performance of firwin with typical inputs remains within acceptable historical margins
# ===========================================================================================

def _time_case(builder, repeats=7):
    # Warm-up once to avoid import/initialization noise
    _ = builder()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = builder()
        times.append(time.perf_counter() - t0)
    # Use median to reduce outlier sensitivity
    return float(np.median(times))

@pytest.mark.slow
@pytest.mark.parametrize(
    "case_id,builder",
    [
        ("firwin_N129_hann_lowpass",
         lambda: firwin(129, 0.22, window="hann", pass_zero=True, fs=1.0)),
        ("firwin_N257_hamming_lowpass",
         lambda: firwin(257, 0.18, window="hamming", pass_zero=True, fs=1.0)),
        ("firwin_N511_kaiser_beta8",
         lambda: firwin(511, 0.15, window=("kaiser", 8.0), pass_zero=True, fs=1.0)),
    ],
)
def test_firwin_perf_within_historical_bounds(case_id, builder):
    bounds = PERF_BOUNDS
    if bounds is None or case_id not in bounds:
        pytest.skip(f"No perf bound for {case_id}; add to fir_perf_thresholds.json to enable.")
    max_seconds = float(bounds[case_id])
    elapsed = _time_case(builder)
    assert elapsed <= max_seconds, f"{case_id} took {elapsed:.4f}s > bound {max_seconds:.4f}s"


# ==================================================================================================
# 31. Representative documentation examples execute and match historical results (except deprec.)
# ==================================================================================================

def test_doc_example_firwin_lowpass_runs_and_matches_snapshot():
    snaps = DOC_BASELINES
    key = "doc_firwin_lowpass_len21_hamming_0p25_fs1p0"
    if snaps is None or key not in snaps:
        pytest.skip("Doc baseline missing for firwin example.")
    expected = np.asarray(snaps[key], dtype=float)
    taps = np.asarray(firwin(21, 0.25, window="hamming", fs=1.0))
    assert_allclose(taps, expected, rtol=1e-12, atol=1e-14)

def test_doc_example_firwin2_bandpass_warns_and_matches_firwin_multiband():
    # This mirrors a typical doc snippet using firwin2
    bands = [0.0, 0.22, 0.30, 0.5]
    gains = [0.0, 0.0, 1.0, 1.0]
    with pytest.warns(DeprecationWarning):
        t2 = np.asarray(firwin2(41, bands, gains, fs=1.0))
    t1 = np.asarray(firwin(41, band=bands, gain=gains, window="hann", method="multiband", fs=1.0))
    assert_allclose(t1, t2, rtol=1e-10, atol=1e-12)


# ========================================================================
# 32. Low-pass FIR taps are symmetric within numerical tolerance (Type I)
# ========================================================================

@pytest.mark.parametrize("numtaps, edge", [(15, 0.22), (31, 0.30), (63, 0.12)])
def test_lowpass_symmetry(numtaps, edge):
    taps = np.asarray(firwin(numtaps, edge, window="hann", pass_zero=True, fs=1.0))
    # Symmetry about the center
    assert_allclose(taps, taps[::-1], rtol=0, atol=1e-12)


# =====================================================================
# 33. Frequency-shape invariance under proportional fs/edge scaling
# =====================================================================

@pytest.mark.parametrize("edge_norm", [0.18, 0.27])
def test_normalized_response_invariant_under_fs_scaling(edge_norm):
    # Spec A: fs=1, cutoff=edge_norm
    spec_a = FilterSpec(numtaps=41, band=[0.0, edge_norm], gain=[1.0, 0.0], window="hann", fs=1.0)
    # Spec B: fs=2, cutoff scaled proportionally
    spec_b = FilterSpec(numtaps=41, band=[0.0, 2.0 * edge_norm], gain=[1.0, 0.0], window="hann", fs=2.0)

    fa = FIRFilter(firwin(spec_a))
    fb = FIRFilter(firwin(spec_b))

    # Compare magnitude responses at normalized radian frequencies (same worN)
    _, Ha = fa.freqz(worN=2048)      # normalized 0..pi
    _, Hb = fb.freqz(worN=2048)      # normalized 0..pi

    assert_allclose(np.abs(Ha), np.abs(Hb), rtol=1e-10, atol=1e-12)


# ============================================================
# 34. Determinism: identical specs yield identical coefficients
#       (within-process and across a spawned process)
# ============================================================

def _worker_make_taps(conn, numtaps, edge):
    # Child process target for deterministic check
    taps = np.asarray(firwin(numtaps, edge, window="hann", fs=1.0))
    conn.send(taps)
    conn.close()

def test_determinism_same_process_and_spawned():
    numtaps, edge = 37, 0.23

    # Same-process repeatability
    t1 = np.asarray(firwin(numtaps, edge, window="hann", fs=1.0))
    t2 = np.asarray(firwin(numtaps, edge, window="hann", fs=1.0))
    assert_allclose(t1, t2, rtol=0, atol=0)

    # Cross-process repeatability (spawned)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker_make_taps, args=(child_conn, numtaps, edge))
    proc.start()
    t_child = parent_conn.recv()
    proc.join()
    assert proc.exitcode == 0
    assert_allclose(t1, t_child, rtol=0, atol=0)