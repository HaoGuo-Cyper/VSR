# Metric source provenance

These files are retained byte-for-byte from the existing VSR experiment. Their
recorded SHA256 values are in `docs/perceptual-provenance.json`.

- `tecogan_metrics.py` comes from [TecoGAN metrics.py](https://github.com/thunil/TecoGAN/blob/bc747a4651c1e08bd0a73bed08c4cb9fac99b4d5/metrics.py), revision `bc747a4651c1e08bd0a73bed08c4cb9fac99b4d5`. See `LICENSE.TecoGAN`.
- `tecogan_networks_basic.py`, `tecogan_pretrained_networks.py` and `tecogan_util.py` preserve TecoGAN's LPIPS reference implementations. The LPIPS source revision recorded by the experiment is [PerceptualSimilarity@082bb24](https://github.com/richzhang/PerceptualSimilarity/tree/082bb24f84c091ea94de2867d34c4544f68e0963). See both `LICENSE.TecoGAN` and `LICENSE.LPIPS`.

The historical `tecogan_alex.pth` and `tecogan_v0.1_alex.pth` calibration binaries
are excluded from this public code import. They remain in the original local
repository and have the same recorded SHA256
`df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0`.
Runtime metric weights are read from the separately linked local `weights/`.
