from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        windows = []

        left_pad = np.zeros(self.window_size // 2)
        right_pad = np.zeros(self.window_size // 2)
        waveform_with_padding = np.concat([left_pad, waveform, right_pad])
        for start in range(0, waveform_with_padding.shape[0] - self.window_size + 1, self.hop_length):
            end = start + self.window_size
            windows.append(waveform_with_padding[start: end])
        return np.stack(windows)
    

class Hann:
    def __init__(self, window_size=1024):
        self.hann_window = scipy.signal.windows.hann(window_size, sym=False)
    
    def __call__(self, windows):
        return self.hann_window * windows



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        spec = np.fft.rfft(windows, n=None)
        spec = np.absolute(spec)
        return spec[:, :self.n_freqs]


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        self.mel_matrix = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=1,
            fmax=8192,
        )
        self.mel_pinv_matrix = np.linalg.pinv(self.mel_matrix)



    def __call__(self, spec):
        mel = spec @ self.mel_matrix.T
        return mel

    def restore(self, mel):
        spec = mel @ self.mel_pinv_matrix.T
        return spec


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        return mel[::-1, :]



class Loudness:
    def __init__(self, loudness_factor):
        self.loudness_factor = loudness_factor


    def __call__(self, mel):
        return self.loudness_factor * mel




class PitchUp:
    def __init__(self, num_mels_up):
        self.k = num_mels_up


    def __call__(self, mel):
        mel_pitched = np.zeros_like(mel)
        mel_pitched[:, self.k:] = mel[:, :-self.k]
        mel_pitched[:, :self.k] = 0
        return mel_pitched



class PitchDown:
    def __init__(self, num_mels_down):
        self.k = num_mels_down

    def __call__(self, mel):
        mel_pitched = np.zeros_like(mel)
        mel_pitched[:, :-self.k] = mel[:, self.k:]
        mel_pitched[:, -self.k:] = 0
        return mel_pitched
        



class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        self.speed_up_factor = speed_up_factor

    def __call__(self, mel):
        indices = np.linspace(0, mel.shape[0] - 1, int(self.speed_up_factor * mel.shape[0]))
        indices = np.round(indices).astype("int")
        return mel[indices, :]
        



class FrequenciesSwap:
    def __call__(self, mel):
        return mel[:, ::-1]



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        self.q = quantile

    def __call__(self, mel):
        value = np.quantile(mel, self.q)
        out = np.copy(mel)
        out[np.where(mel < value)] = 0
        return out



class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

