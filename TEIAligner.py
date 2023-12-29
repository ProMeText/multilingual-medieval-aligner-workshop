from bertalign import Bertalign
import lxml.etree as etree
import bertaling.tokenization as tokenization
import bertalign.utils as utils
import json



class TEIAligner():
    """
    L'aligneur, qui prend des fichiers TEI en entrée (tokénisés?)
    """
    def __init__(self, files:list, tokenize=False):
        self.tei_ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        with open("bertalign/delimiters.json", "r") as input_json:
            dictionary = json.load(input_json)
        single_tokens_punctuation = [punct for punct in dictionary['punctuation'] if len(punct) == 1]
        multiple_tokens_punctuation = [punct for punct in dictionary['punctuation'] if len(punct) != 1]
        single_token_punct = "".join(single_tokens_punctuation)
        multiple_tokens_punct = "|".join(multiple_tokens_punctuation)
        punctuation_subregex = f"({multiple_tokens_punct}|[{single_token_punct}])"
        tokens_subregex = "(" + " | ".join(dictionary['word_delimiters']) + ")"
        self.parsed_files = []
        if tokenize:
            tokenizer = tokenization.Tokenizer(regularisation=True)
            for file in files:
                tokenizer.tokenisation(path=file, punctuation_regex=punctuation_subregex)
                regularized_file = file.replace('.xml','.regularized.xml')
                utils.pretty_print_xml_tree(regularized_file)
                tokenizer.subsentences_tokenisation(path=regularized_file, delimiters=tokens_subregex)
                self.parsed_files.append(tokenizer.tokenized_tree)
        else:
            self.parsed_files = [etree.parse(file) for file in files]
        self.main_file = self.parsed_files[0]
        self.files = self.parsed_files[1:]
    
    
    def alignementMultilingue(self):
        source_tokens, target_tokens = list(), list()
        for text in self.files:
            for phrase in text.xpath("descendant::tei:phr", namespaces=self.tei_ns):
                target_tokens.append([token.text for token in phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]", namespaces=self.tei_ns)])

            for phrase in self.main_file.xpath("descendant::tei:phr", namespaces=self.tei_ns):
                source_tokens.append([token.text for token in phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]", namespaces=self.tei_ns)])
            aligner = Bertalign(source_tokens, target_tokens)
            aligner.align_sents()
            aligner.print_sents()
            
            
    def inject_sents(self, results, source_zip, target_zip):
        """
        Avec cette fonction on récupère l'alignement sur le texte et on le réinjecte dans le fichier TEI
        """
        pass
    
    def alignement_de_structures(self):
        """
        On se sert de l'alignement sémantique pour aligner des structures sur un document cible à partir 
        d'un document source. Alignement puis identification de la borne supérieure de la structure (division, titre)
        On se servira d'un calcul de similarité pour identifier précisément la fin de la division dans le document cible
        """
        pass
    
    
    
if __name__ == '__main__':
    # TODO: intégrer les noeuds non w|pc pour ne pas perdre cette information.
    file_list = ["/projects/users/mgillele/alignment/bertalign/text+berg/local_data/Val_S.citable.xml",
                 "/projects/users/mgillele/alignment/bertalign/text+berg/local_data/Rome_W.xml"]
    Aligner = TEIAligner(file_list, tokenize=True)
    Aligner.alignementMultilingue()