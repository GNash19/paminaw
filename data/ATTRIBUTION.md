# Data attribution and licences

The **code** in this repository is MIT-licensed (see `LICENSE`). The data files
below have their own licences.

## FLEURS: audio clips and reference transcriptions (CC BY 4.0)

- **Source:** FLEURS, Conneau, A., Ma, M., Khanuja, S., et al. *FLEURS:
  Few-shot Learning Evaluation of Universal Representations of Speech.* IEEE
  SLT 2022, pp. 798–805 (2023). Distributed as
  [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Revision used:** Hub commit `70bb2e84b976b7e960aa89f1c648e09c59f894dd`,
  test splits of `fil_ph` (Tagalog) and `ceb_ph` (Cebuano).
- **What we redistribute:**
  - `subset_manifest.csv`: the FLEURS `id`, `transcription`, `raw_transcription`,
    `num_samples` and `gender` fields for 100 + 100 test recordings.
  - The release asset `paminaw_clips_v1.zip`, which holds those 200 WAV files.
- **Changes made:**
  - We selected a subset with `scripts/make_subset.py`, seed 20260814.
  - We added the columns `lang`, `draw_order`, `idx`, `filename` and `duration_s`.
  - The audio bytes are **unchanged**.
- FLEURS is read speech from prepared prompts. It contains no patient or
  clinical recordings, and this workshop never uses any.

## Precomputed model outputs: `precomputed/` (CC BY-NC 4.0)

- **Source:** Golosino, N., Morales, M. N., & Adlaon, K. M. (2026). *Data and
  Code Release for "Benchmarking Multilingual ASR for Tagalog and Cebuano
  Across Scoring Conventions and Language Coverage"* (NLPIR 2026). Zenodo,
  <https://doi.org/10.5281/zenodo.23010849> (version 2, record 23015266),
  licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- **What we redistribute:** the hypotheses for the 200 subset clips only, from
  three of the paper's runs. They are **unnormalised**, exactly as the models
  produced them. `precomputed/models.json` records each run's model revision,
  precision and seed. They were extracted with `scripts/extract_precomputed.py`.
- **Licence:** every file in `precomputed/` is CC BY-NC 4.0, the licence of
  the Zenodo record. In particular, the outputs of **MMS-1B-all** and
  **SeamlessM4T v2 Large** are for **non-commercial research use**, consistent
  with those models' own CC BY-NC 4.0 licences.

| System | Model licence (Hugging Face) | Outputs here |
|---|---|---|
| `openai/whisper-large-v3` | Apache-2.0 | CC BY-NC 4.0 (Zenodo record licence) |
| `facebook/mms-1b-all` | CC BY-NC 4.0 | CC BY-NC 4.0, non-commercial research use |
| `facebook/seamless-m4t-v2-large` | CC BY-NC 4.0 | CC BY-NC 4.0, non-commercial research use |

### Exception: `precomputed/whisper-small.csv` (CC BY 4.0)

This file is **not** from the paper. The facilitators generated it on a CPU with
`scripts/precompute_large.py --system whisper-small`; `whisper-small.json` records the revision
and settings. Lab 3 uses it when an attendee skipped Lab 2. It is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), like the FLEURS audio it transcribes.

The models that attendees run live in the notebooks are `openai/whisper-small`
(Apache-2.0) and, on GPU runtimes, `facebook/mms-1b-all` (CC BY-NC 4.0). They are
downloaded from the Hugging Face Hub and are not redistributed here.
