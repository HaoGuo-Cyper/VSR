# Protocol33 window and asset preparation audit — 2026-11-05

**The metadata contract passes for all 1,673 selected clips.** The three new Base configurations keep the original 1,641 training clips and the identical 32 validation clips, in the same order, with data/window seed 20261102. Their training seeds are 20261102/03/04. The full manifest remains train1641/val201/test211; no split was changed.

## Dense sampling of the same source window

Old9 frames × stride4 and new33 frames × stride1 both require exactly33 source frames. Every original manifest record already passes that header eligibility (2,053/2,053); the audit-only subset needing129 source frames for33/stride4 is irrelevant to the new protocol.

For every selected training/validation record, an independent computation from the sealed source header and the unchanged window hash establishes:

- Same record/source ID, split and recorded source-file SHA.
- Same window hash seed, source start and inclusive source end.
- Old nine source indices equal every fourth element of the new33 consecutive indices.
- Same source dimensions, aspect-cover resize, centered128×224 crop and bicubic ×4 parameters.
- All three configurations choose the identical train/val IDs; only the training RNG seed differs.

The all-1,673 mapping digest is `a5acda5cb5592e0224d3cee2fc4c6dc6ffd9d6f18d5783f76fdaae91c0f877ab`. The JSON companion specifies its canonical format and gives an example. Original videos were not decoded or rehashed by this metadata audit; the new preparation's `decode_clip` must still verify raw source bytes and successful decoding. Old output FPS was original/4; new FPS is original. These are denser real samples, not interpolated frames.

## Reusable text-only cache

Source: `/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/fresh_text_embeddings.pt`.

- File SHA256: `1dd219cde3005b4d16febaad67ad87e79c1403ae9bdaae69f2723ab9cbf239c2`; 4,196,043 bytes.
- Payload independently inspected on CPU with `weights_only=True`: one fixed-caption entry, finite frozen BF16`[1,512,4096]`; CUDA remained uninitialized.
- Raw tensor bytes SHA256: `5b91234021648cdcd628242017f43152b72e4b0348b98cf44f0480746afd84ec`.
- Stage2 dtype/shape-header+tensor digest: `f99e3246097b8b2052c9e0f1b8ab1e9b08d02cae42cd899e77dae3f20b9bd127`. This uses a different domain and must not be confused with the raw tensor digest.
- Neighbor originating config SHA256: `fb03abf27dfa4ee393310d99743310855913f864f19080f2fd83b8c079145639`.
- Original source config SHA256: `597e99b658f60684983966d8f3287f999bbb04285b0667f6ac077c062261f6d7`; it matches the origin's recorded `_sha256`.
- Same pinned official Wan repo/revision/local model root; text-audit SHA `c84d3ec52e41914cd700c91b47196c739a90c593a86db87316b18b0404dfb3ee`.

This fixed text embedding does not depend on the number or spacing of video frames. Only text is reused. Old RGB/VAE latent caches remain incompatible: geometry changes from3 to9 latent positions, and stride changes4→1. Missing/tampered text must fail without loading T5 or recomputing embeddings. Base-only auxiliary preparation must not construct DINO or RAFT; `variants` and `preparation_variants` are both`[Base]` and auxiliary weights are zero in all three configs.

## Memory and storage

For1,673 clips, float32 paired RGB tensors alone would occupy 36.487 GiB; including valid masks gives 37.962 GiB. That exceeds the host's approximately29.95 GiB total RAM, so the legacy eager `train.prepare` path must not be used. Its fresh branch also unconditionally calls `encode_text` before RGB accumulation.

Streaming one RGB clip at a time and retaining normalized latent pairs is feasible in host RAM: all latent pairs are approximately0.804 GiB, plus a bounded RGB LRU and other runtime state. The estimate for one disk cache including RGB and both per-clip/top-level latent copies is 38.096 GiB. Three entirely independent data caches would be 114.287 GiB; sharing the newly sealed cache avoids that duplication. Container/metadata/preview/checkpoint overhead is additional.

Snapshot: 691,675,930,624 disk bytes available on the project filesystem and 25,637,146,624 host memory bytes available. These are transient observations, not reservations. No GPU memory or training capacity was measured in this audit.

## Source and runtime requirements

All1,673 old per-clip bindings match the unchanged current encoder hashes:

- `vsr/data.py`: `654bd03809cca1649cbd618b145751b723c1ea2a4e68738a53d4c90dddda82a0`
- `vsr/preflight.py`: `7e291163598c4fd1c73cd38567603381eeaaccf0c1332c4d81beb041eba9ca40`

The new streaming preparation/queue/configs need their own frozen source bindings; they cannot inherit an old cache's completion seal. New caches must live under the actual new checkout's `runs/`, with that allowed root passed explicitly to lazy/cache validators. Runtime source must be committed/clean and recorded before execution. Each of the three Base models/optimizers must initialize freshly from the official base, while sharing only the new data cache. This audit wrote only these text reports and did not launch GPU work, change configs/core code or publish anything.
