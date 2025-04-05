# AQUILIGN -- Mutilingual aligner and collator

[![codecov](https://codecov.io/github/ProMeText/Aquilign/graph/badge.svg?token=TY5HCBOOKL)](https://codecov.io/github/ProMeText/Aquilign)


This repo contains a set of scripts to align (and soon collate) a multilingual medieval corpus. Its designers are Matthias Gille Levenson, Lucence Ing and Jean-Baptiste Camps.  

It is based on a fork of the automatic multilingual sentence aligner Bertalign.

The scripts relies on a prior phase of text segmentation at syntagm level using regular expressions or bert-based segmentation to match grammatical syntagms and produce a more precise alignment.

## Use

`python3 main.py -o lancelot -i data/extraitsLancelot/ii-48/ -mw data/extraitsLancelot/ii-48/fr/micha-ii-48.txt -d 
cuda:0 -t bert-based` to perform alignment with our bert-based segmenter, choosing Micha edition as base witness,
on the GPU. The results will be saved in `result_dir/lancelot`

`python3 main.py --help` to print help.

Files must be sorted by language, using the ISO_639-1 language code as parent directory name (`es`, `fr`, `it`, `en`, etc).
## Citation

Gille Levenson, M., Ing, L., & Camps, J.-B. (2024, November 1). Textual Transmission without Borders: Multiple Multilingual Alignment and Stemmatology of the “Lancelot en prose” (Medieval French, Castilian, Italian). Computational Humanities Research 2024. https://enc.hal.science/hal-04759151

```
@inproceedings{gillelevenson_TextualTransmissionBorders_2024,
  title = {Textual {{Transmission}} without {{Borders}}: {{Multiple Multilingual Alignment}} and {{Stemmatology}} of the "{{Lancelot}} En Prose" ({{Medieval French}}, {{Castilian}}, {{Italian}})},
  shorttitle = {Textual {{Transmission}} without {{Borders}}},
  author = {Gille Levenson, Matthias and Ing, Lucence and Camps, Jean-Baptiste},
  date = {2024-11-01},
  url = {https://enc.hal.science/hal-04759151},
  urldate = {2024-11-11},
  abstract = {This study focuses on the problem of multilingual medieval text alignment, which presents specific challenges, due to the absence of modern punctuation in the texts and the non-standard forms of medieval languages. In order to perform the alignment of several witnesses from the multilingual tradition of the prose Lancelot, we first develop an automatic text segmenter based on BERT and then align the produced segments using Bertalign. This alignment is then used to produce stemmatological hypotheses, using phylogenetic methods. The aligned sequences are clustered independently by two human annotators and a clustering algorithm (DBScan), and the resulting variant tables submitted to maximum parsimony analysis, in order to produce trees. The trees are then compared and discussed in light of philological knowledge. Results tend to show that automatically clustered sequences can provide results comparable to those of human annotation.},
  eventtitle = {Computational {{Humanities Research}} 2024},
  langid = {english},
  file = {/home/mgl/Bureau/Travail/Bibliotheque_zoteros/storage/HRW4Z63I/Levenson et al. - 2024 - Textual Transmission without Borders Multiple Multilingual Alignment and Stemmatology of the Lance.pdf}
}
```


## Licence

This fork is released under the [GNU General Public License v3.0](./LICENCE)

## Funding

This work benefited́ from national funding managed by the Agence Nationale de la Recherche under the Investissements d'avenir programme with the reference ANR-21-ESRE-0005 (Biblissima+). 

Ce travail a bénéficié́ d'une aide de l’État gérée par l'Agence Nationale de la Recherche au titre du programme d’Investissements d’avenir portant la référence ANR-21-ESRE-0005 (Biblissima+) 

![image](https://github.com/user-attachments/assets/915c871f-fbaa-45ea-8334-2bf3dde8252d)

