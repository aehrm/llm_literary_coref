# German Full Novel Corpus (GerFuN)

This repository contains a corpus of German literary novels annotated for mentions, coreference, and attributes on mention- and entity-level.

The annotation guidelines used for this corpus are available at [Zenodo (zenodo.org/records/20364509)](https://zenodo.org/records/20364509).

We provide both the original INCEpTION export files (`.xmi` in `annotated_xmi/`), and token-aligned tabular outputs (`.tsv` in `annotated_tsv/`). See `convert_annotations.py` for the conversion script.


To estimate coreference annotation reliability, the first 2,000 words of each novel were independently annotated by two annotators, coded as `ahi` and `ohe` (see folders `iaa`). On clustering metrics, IAA is 87.19% LEA and 88.65% CoNLL-F1), as calculated by `calculate_iaa.py`. See `evaluation_reports/iaa_report.txt` and the [JCLS publication](https://doi.org/10.26083/tuda-7983) for more details.
