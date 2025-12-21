#!/bin/sh
set -e

uv run python convert_annotations.py --source_tsv sources/Fischer_Gustav_chapter_2_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ahi_Fischer_Gustav_chapter_2_first_2000.xmi --output_tsv annotated_tsv/iaa/ahi_Fischer_Gustav_chapter_2_first_2000.tsv
uv run python convert_annotations.py --source_tsv sources/Heimburg_Trudchen_chapter_1_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ahi_Heimburg_Trudchen_chapter_1_first_2000.xmi --output_tsv annotated_tsv/iaa/ahi_Heimburg_Trudchen_chapter_1_first_2000.tsv
uv run python convert_annotations.py --source_tsv sources/Kürnberger_Amerika_chapter_2_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ahi_Kürnberger_Amerika_chapter_2_first_2000.xmi --output_tsv annotated_tsv/iaa/ahi_Kürnberger_Amerika_chapter_2_first_2000.tsv
uv run python convert_annotations.py --source_tsv sources/Wolff_Wildfangrecht_chapter_2_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ahi_Wolff_Wildfangrecht_chapter_2_first_2000.xmi --output_tsv annotated_tsv/iaa/ahi_Wolff_Wildfangrecht_chapter_2_first_2000.tsv
uv run python convert_annotations.py --source_tsv sources/Fischer_Gustav_chapter_2_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ohe_Fischer_Gustav_chapter_2_first_2000.xmi --output_tsv annotated_tsv/iaa/ohe_Fischer_Gustav_chapter_2_first_2000.tsv
uv run python convert_annotations.py --source_tsv sources/Heimburg_Trudchen_chapter_1_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ohe_Heimburg_Trudchen_chapter_1_first_2000.xmi --output_tsv annotated_tsv/iaa/ohe_Heimburg_Trudchen_chapter_1_first_2000.tsv
uv run python convert_annotations.py --source_tsv sources/Kürnberger_Amerika_chapter_2_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ohe_Kürnberger_Amerika_chapter_2_first_2000.xmi --output_tsv annotated_tsv/iaa/ohe_Kürnberger_Amerika_chapter_2_first_2000.tsv
uv run python convert_annotations.py --source_tsv sources/Wolff_Wildfangrecht_chapter_2_first_2000.tsv --annotation_xmi annotated_xmi/iaa/ohe_Wolff_Wildfangrecht_chapter_2_first_2000.xmi --output_tsv annotated_tsv/iaa/ohe_Wolff_Wildfangrecht_chapter_2_first_2000.tsv

uv run python convert_annotations.py --source_tsv sources/Fischer_Gustav.tsv --annotation_dir annotated_xmi/Fischer_Gustav --output_tsv annotated_tsv/Fischer_Gustav.tsv
uv run python convert_annotations.py --source_tsv sources/Goethe_Wahlverwandtschaften.tsv --annotation_dir annotated_xmi/Goethe_Wahlverwandtschaften --output_tsv annotated_tsv/Goethe_Wahlverwandtschaften.tsv
uv run python convert_annotations.py --source_tsv sources/Heimburg_Trudchen.tsv --annotation_dir annotated_xmi/Heimburg_Trudchen --output_tsv annotated_tsv/Heimburg_Trudchen.tsv
uv run python convert_annotations.py --source_tsv sources/Kürnberger_Amerika.tsv --annotation_dir annotated_xmi/Kürnberger_Amerika --output_tsv annotated_tsv/Kürnberger_Amerika.tsv
uv run python convert_annotations.py --source_tsv sources/Wolff_Wildfangrecht.tsv --annotation_dir annotated_xmi/Wolff_Wildfangrecht --output_tsv annotated_tsv/Wolff_Wildfangrecht.tsv
