# AQUILIGN -- Mutilingual aligner and collator

[![codecov](https://codecov.io/github/ProMeText/Aquilign/graph/badge.svg?token=TY5HCBOOKL)](https://codecov.io/github/ProMeText/Aquilign)


This repo contains a set of scripts to align (and soon collate) a multilingual medieval corpus. Its designers are Matthias Gille Levenson, Lucence Ing and Jean-Baptiste Camps.  

It is based on a fork of the automatic multilingual sentence aligner Bertalign.

The scripts relies on a prior phase of text segmentation at syntagm level using regular expressions or bert-based segmentation to match grammatical syntagms and produce a more precise alignment.

## Training the segmenter

The segmenter we use is based on a Bert AutoModelForTokenClassification that is trainable. 

Example of use: 

`python3 train_tokenizer.py -m google-bert/bert-base-multilingual-cased  
-t ../Multilingual_Aegidius/data/segmentation_data/split/multilingual/full.train.json 
-d ../Multilingual_Aegidius/data/segmentation_data/split/multilingual/full.dev.json 
-e ../Multilingual_Aegidius/data/segmentation_data/split/multilingual/full.eval.json 
-ep 100 
-b 128 
--device cuda:0 
-bf16 
-n multilingual_model 
-s 2 
-es 10`

For finetuning a multilingual model from the `bert-base-multilingual-cased` model, on 100 epochs, a batch size of 128,
on the GPU, using bf16 precision, saving the model every two epochs and with and early stopping value of 10.

The training data must follow the following structure and will be validated against a specific JSON schema.

```JSON
{"metadata": 
  {
    "lang": ["la", "it", "es", "fr", "en", "ca", "pt"],
    "centuries": [13, 14, 15, 16], "delimiter": "£"
  },
"examples": 
    [
      {"example": "que mi padre me diese £por muger a un su fijo del Rey", 
        "lang": "es"},
      {"example": "Per fé, disse Lion, £i v’andasse volentieri, £ma i vo veggio £qui", 
        "lang": "it"}
    ]
}
```
The metadata is used for describing the corpus and will be parsed in search for the delimiter. It is the only mandatory 
information.

We recommend using the ISO codes for the target languages. 
The codes must match the language codes that are in the [`aquilign/preproc/delimiters.json`](aquilign/preproc/delimiters.json) file, used for the
regexp tokenization that can be used as a baseline. 

## Use of the aligner

`python3 main.py -o lancelot -i data/extraitsLancelot/ii-48/ -mw data/extraitsLancelot/ii-48/fr/micha-ii-48.txt -d 
cuda:0 -t bert-based` to perform alignment with our bert-based segmenter, choosing Micha edition as base witness,
on the GPU. The results will be saved in `result_dir/lancelot`

`python3 main.py --help` to print help.

Files must be sorted by language, using the ISO_639-1 language code as parent directory name (`es`, `fr`, `it`, `en`, etc).
## Citation

Gille Levenson, M., Ing, L., & Camps, J.-B. (2024). Textual Transmission without Borders: Multiple Multilingual Alignment and Stemmatology of the ``Lancelot en prose’’ (Medieval French, Castilian, Italian). In W. Haverals, M. Koolen, & L. Thompson (Eds.), Proceedings of the Computational Humanities   Research Conference 2024 (Vol. 3834, pp. 65–92). CEUR. https://ceur-ws.org/Vol-3834/#paper104


```
@inproceedings{gillelevenson_TextualTransmissionBorders_2024a,
  title = {Textual {{Transmission}} without {{Borders}}: {{Multiple Multilingual Alignment}} and {{Stemmatology}} of the ``{{Lancelot}} En Prose'' ({{Medieval French}}, {{Castilian}}, {{Italian}})},
  shorttitle = {Textual {{Transmission}} without {{Borders}}},
  booktitle = {Proceedings of the {{Computational Humanities}}   {{Research Conference}} 2024},
  author = {Gille Levenson, Matthias and Ing, Lucence and Camps, Jean-Baptiste},
  editor = {Haverals, Wouter and Koolen, Marijn and Thompson, Laure},
  date = {2024},
  series = {{{CEUR Workshop Proceedings}}},
  volume = {3834},
  pages = {65--92},
  publisher = {CEUR},
  location = {Aarhus, Denmark},
  issn = {1613-0073},
  url = {https://ceur-ws.org/Vol-3834/#paper104},
  urldate = {2024-12-09},
  eventtitle = {Computational {{Humanities Research}} 2024},
  langid = {english},
  file = {/home/mgl/Bureau/Travail/Bibliotheque_zoteros/storage/CIH7IAHV/Levenson et al. - 2024 - Textual Transmission without Borders Multiple Multilingual Alignment and Stemmatology of the ``Lanc.pdf}
}

```


## Licence

This fork is released under the [GNU General Public License v3.0](./LICENCE)

## Funding

This work benefited́ from national funding managed by the Agence Nationale de la Recherche under the Investissements d'avenir programme with the reference ANR-21-ESRE-0005 (Biblissima+). 

Ce travail a bénéficié́ d'une aide de l’État gérée par l'Agence Nationale de la Recherche au titre du programme d’Investissements d’avenir portant la référence ANR-21-ESRE-0005 (Biblissima+) 

![image](https://github.com/user-attachments/assets/915c871f-fbaa-45ea-8334-2bf3dde8252d)

