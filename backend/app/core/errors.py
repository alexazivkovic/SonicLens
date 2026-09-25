"""Errors raised while loading or analysing audio. The API maps them to 4xx responses."""


class AudioError(Exception):
    """Base class for problems with the input audio (not bugs in SonicLens)."""


class AudioDecodeError(AudioError):
    """The file could not be decoded as audio."""


class AudioTooShortError(AudioError):
    """The audio is shorter than the minimum analysable duration."""


class SilentAudioError(AudioError):
    """The audio contains no audible signal."""
