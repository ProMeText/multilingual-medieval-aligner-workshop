# Mutilingual collator


This repo contains a set of scripts to align and collate a multilingual medieval corpus. Its designers are Matthias Gille Levenson, Lucence Ing and Jean-Baptiste Camps.  

It is based on a fork of the automatic multilingual sentence aligner Bertalign.

The scripts relies for now on a prior phase of text segmentation at syntagm level using regular expressions to match grammatical syntagms and produce a more precise alignment.

## Use

`python3 python/multiple_macro_alignment.py data/extraitsLancelot/micha_ii-48.txt  data/extraitsLancelot/sommer_tome_4-ii-48.txt data/extraitsLancelot/lanzarote-ii-48.txt data/extraitsLancelot/lancellotto-ii-48.txt data/extraitsLancelot/inc-ii-48.txt data/extraitsLancelot/fr111-ii-48.txt data/extraitsLancelot/fr751-ii-48.txt 0 lancelot_1 True`

## Citation

Lei Liu & Min Zhu. 2022. Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts, *Digital Scholarship in the Humanities*. [https://doi.org/10.1093/llc/fqac089](https://doi.org/10.1093/llc/fqac089).


## Licence

This fork is released under the [GNU General Public License v3.0](./LICENCE)

