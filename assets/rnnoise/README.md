# RNNoise model for `audio.denoise` (FFmpeg `arnndn`)

`sh.rnnn` is the "somnolent-hogwash" model from
[GregorR/rnnoise-models](https://github.com/GregorR/rnnoise-models)
(`somnolent-hogwash-2018-09-01/sh.rnnn`), trained for the "recording noise /
speech signal" case — i.e. general voice recordings with background noise,
which matches this capability's use case.

Per that repo's `README.md`: "With the exception of the tools/ directory and
this file, none of this work is creative and thus none of it is subject to
copyright." No separate LICENSE file is published; GitHub's license detector
reports none. Treat as public-domain-equivalent per the author's stated intent.

Loaded at `/usr/share/rnnoise/model.rnnn` in the Docker image
(`DEFAULT_RNNOISE_MODEL` in `app/tools/ffmpeg.py`).
