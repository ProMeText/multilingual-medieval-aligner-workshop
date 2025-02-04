## AQUILIGN -- Mutilingual aligner and collator 
### Corpus for Multilingual Alignment Project - Phase II 

### Overview
This corpus has been developed for the second phase of the Multilingual Alignment Project, expanding on our initial work detailed in the article [Textual Transmission without Borders](https://2024.computational-humanities-research.org/papers/paper104/). The first phase focused on aligning textual data across three medieval languages—Castilian, French, and Italian. In Phase II, we extend our scope to include four additional languages: Portuguese, Catalan, Latin, and English, broadening our linguistic and cultural reach.

### Purpose
The primary purpose of this expanded corpus is to train machine learning models for text segmentation— a critical task that involves dividing text into linguistically meaningful units. This capability is crucial for enhancing the accessibility and analysis of historical texts. We aimed to collect prose texts from the 13th to the 15th centuries, potentially extending into the mid-16th century, selected for their thematic variety to ensure a rich dataset that supports robust model training.


### Data Collection Variability Across Languages

The data collection process involved obtaining texts in both TXT and XML formats, with XML files subsequently converted to TXT. All texts underwent a cleaning process using regular expressions to ensure consistency and usability across different languages.

For some languages, like Middle French, the acquisition process was quite straightforward. Cleaned texts in both TXT and XML formats were readily available from the [BFM Corpus](https://gitlab.huma-num.fr/bfm/bfm-textes-diffusion/-/tree/main/TXT?ref_type=heads), which served as the primary source. This ease of access significantly streamlined our workflow for French texts.

In contrast, for Portuguese, English, and Italian, while some texts could be sourced from specific corpora that provide XML downloads, the effort required varied. These sources included the [CTA](https://teitok.clul.ul.pt/teitok/cta/index.php?action=home) for Portuguese, [LAEME](http://amc.ppls.ed.ac.uk/laeme/texts.html) for English, and [Biblioteca Italiana](http://www.bibliotecaitaliana.it/percorsi/19) for Italian. These resources were instrumental in supplying structured data suitable for our analysis but required more intensive data preparation compared to the French texts.

However, accessibility varied significantly across languages. In some instances, corpora like the OVI for Italian and the CICA for Catalan existed, but the texts were not available for public consultation or download, posing additional challenges to data collection. In these cases, we needed to recover texts through critical editions or by scraping from HTML files. This variability in accessibility necessitated a flexible approach to sourcing and cleaning data, depending on the availability and condition of the texts within each corpus.

Overall, the data collection highlighted the considerable variability in the ease of data acquisition across languages, affecting the overall efficiency and methodology of our research.


### Application Form
An application form was designed to streamline the collection process, allowing for the insertion, organization, and storage of texts. This ensures that all textual sources are well-documented and traceable. The form, along with the code and a CSV file compiling all relevant information, can be accessed via the links provided:
- [Application Form](#)
- [Code Repository](https://github.com/carolisteia/mulada)
- [Compiled Data CSV](https://github.com/carolisteia/mulada/blob/main/data.csv)


### Licensing
The texts within this corpus are available under the [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. This license allows users to adapt, remix, and build upon the work non-commercially, as long as they credit the authors and license their new creations under the identical terms. Full details and citations of all sources can be found in our corpus documentation, specifically under the 'sources' and 'corpus' columns of the [compiled data CSV](https://github.com/carolisteia/mulada/blob/main/data.csv).


### Conclusion and Future Directions
As we continue to expand and refine the Multilingual Alignment Project, our objectives include incorporating additional languages and further enhancing our text segmentation models. The next phase will focus on improving the machine learning techniques employed and exploring diverse historical periods and textual forms. We eagerly anticipate ongoing collaborations within the scholarly community to broaden the reach and impact of our work.

### Support and Documentation
For additional support or further documentation regarding the Multilingual Alignment Project, please contact our team directly via [email](mailto:carolina.macedo@chartes.psl.eu).
