from bertalign import Bertalign
import sys
import lxml.etree as etree

class TEIAligner():
    """
    L'aligneur, qui prend des fichiers TEI en entrée (tokénisés?)
    """
    def __init__(self, files:list):
        self.tei_ns = {'tei': {'http://www.tei-c.org/ns/1.0'}}
        self.parsed_files = [etree.parse(file) for file in files]
        self.main_file = self.parsed_files[0]
        self.files = self.parsed_files[1:]
    
    
    def alignementMultilingue(self):
        for text in self.files[1:]:
            self.TEIAlign(self.main_file, text)
    
    def TEIAlign(self, source, target):
        context_div = "tei:div[@type='chapter']/descendant::tei:p[not(ancestor::tei:div[@type='glose'])]"
        for division in source.xpath(context_div, namespaces=self.tei_ns):
            division_n = division.xpath("@n")
            source_as_tokens = division.xpath("descendant::tei:node()[self::tei:w or self::tei:pc]", namespaces=self.tei_ns)
            source_as_ids = division.xpath("descendant::tei:node()[self::tei:w or self::tei:pc]/@xml:id", namespaces=self.tei_ns)
            source_zip_token_ids = list(zip(source_as_tokens, source_as_ids))
            target_as_tokens = target.xpath(f"tei:div[@type='chapter'][@='{division_n}']/descendant::tei:p[not(ancestor::tei:div[@type='glose'])]/descendant::tei:node()[self::tei:w or self::tei:pc]", namespaces=self.tei_ns)
            target_as_ids = target.xpath(f"tei:div[@type='chapter'][@='{division_n}']/descendant::tei:p[not(ancestor::tei:div[@type='glose'])]/descendant::tei:node()[self::tei:w or self::tei:pc]/@xml:id", namespaces=self.tei_ns)
            target_zip_token_ids = list(zip(target_as_tokens, target_as_ids))
            source_as_text = ' '.join(source_as_tokens)
            target_as_text = ' '.join(target_as_tokens)
            aligner = Bertalign(source_as_text, target_as_text)
            aligner.align_sents()
            self.inject_sents(aligner.results, source_zip_token_ids, target_zip_token_ids)
        
    def inject_sents(self, results, source_zip, target_zip):
        """
        Avec cette fonction on récupère l'alignement sur le texte et on le réinjecte dans le fichier TEI
        """
        
    
    def alignement_de_structures(self):
        """
        On se sert de l'alignement sémantique pour aligner des structures sur un document cible à partir 
        d'un document source. Alignement puis identification de la borne supérieure de la structure (division, titre)
        On se servira d'un calcul de similarité pour identifier précisément la fin de la division dans le document cible
        """
        pass
    
    
    
if __name__ == '__main__':
    file_list = []
    Aligner = TEIAligner(file_list)