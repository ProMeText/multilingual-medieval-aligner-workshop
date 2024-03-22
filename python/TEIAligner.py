import argparse

from bertalign.bertalign import Bertalign
import lxml.etree as etree
import bertalign.tokenization as tokenization
import bertalign.utils as utils
import json
import traceback
import tabulate

class TEIAligner():
    """
    L'aligneur, qui prend des fichiers TEI en entrée et les tokénise
    """
    def __init__(self, files_path:dict, tokenize):
        self.tei_ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        with open("bertalign/delimiters.json", "r") as input_json:
            dictionary = json.load(input_json)
        single_tokens_punctuation = [punct for punct in dictionary['punctuation'] if len(punct) == 1]
        multiple_tokens_punctuation = [punct for punct in dictionary['punctuation'] if len(punct) != 1]
        single_token_punct = "".join(single_tokens_punctuation)
        multiple_tokens_punct = "|".join(multiple_tokens_punctuation)
        punctuation_subregex = f"({multiple_tokens_punct}|[{single_token_punct}])"
        tokens_subregex = "(" + " | ".join(dictionary['word_delimiters']) + ")"
        self.target_parsed_files = {}
        self.main_parsed_file = None
        files = files_path['target_files']
        main_file = files_path['main_file']
        # Faut-il tokéniser le document (mots, syntagmes) ?
        if tokenize:
            print("Tokenizing")
            tokenizer = tokenization.Tokenizer(regularisation=True)
            tokenizer.tokenisation(path=main_file, punctuation_regex=punctuation_subregex)
            regularized_file = main_file.replace('.xml', '.regularized.xml')
            utils.pretty_print_xml_tree(regularized_file)
            print("Word tokenization done.")
            tokenizer.subsentences_tokenisation(path=regularized_file, delimiters=tokens_subregex)
            self.main_file = (main_file, tokenizer.tokenized_tree)
            for file in files:
                tokenizer.tokenisation(path=file, punctuation_regex=punctuation_subregex)
                print("Word tokenization done.")
                regularized_file = file.replace('.xml','.regularized.xml')
                utils.pretty_print_xml_tree(regularized_file)
                tokenizer.subsentences_tokenisation(path=regularized_file, delimiters=tokens_subregex)
                self.target_parsed_files[file] = tokenizer.tokenized_tree
            print("Done")
        else:
            self.target_parsed_files = {file: etree.parse(file) for file in files}
            self.main_file = (main_file, etree.parse(main_file))

    def create_alignment_table(self, file_a, file_b):
        tree_a = etree.parse(file_a)
        tree_b = etree.parse(file_b)
        table = []
        for book in tree_a.xpath("descendant::tei:div[@type='livre']", namespaces=self.tei_ns):
            book_n = book.xpath("@n")[0]
            print(book_n)
            phrases_tree_b = tree_b.xpath("descendant::tei:phr", namespaces=self.tei_ns)
            id_phrases_tree_b = tree_b.xpath("descendant::tei:phr/@xml:id", namespaces=self.tei_ns)
            ids_and_phrases_b = {id:phrase for id, phrase in list(zip(id_phrases_tree_b, phrases_tree_b))}
            for part in book.xpath("descendant::tei:div[@type='partie']", namespaces=self.tei_ns):
                part_n = part.xpath("@n")[0]
                print(part_n)
                for chapter in part.xpath("descendant::tei:div[@type='chapitre']", namespaces=self.tei_ns):
                    chapter_n = chapter.xpath("@n")[0]
                    print(chapter_n)
                    previous_phrase = ""
                    for index, phrase in enumerate(chapter.xpath("descendant::tei:phr", namespaces=self.tei_ns)):
                        corresponding_ids = phrase.xpath('@corresp')[0]
                        intermed_list = [book_n, part_n, chapter_n]
                        phrase_as_string = ' '.join([token.text for token in phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]", namespaces=self.tei_ns)])
                        intermed_list.append(phrase_as_string)
                        individual_idents = [ident.replace('#', '') for ident in corresponding_ids.split()]
                        phrase_text_b = ""
                        for ident in individual_idents:
                            try:
                                corresponding_phrase = ids_and_phrases_b[ident.replace('#', '')]
                                corresponding_phrase_as_string = ' '.join([token.text for token in corresponding_phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]", namespaces=self.tei_ns)])
                            except KeyError:
                                print(f"Issue with |{ident}|")
                                corresponding_phrase_as_string = ""
                            phrase_text_b += f" {corresponding_phrase_as_string}"
                        if previous_phrase == phrase_text_b:
                            intermed_list.append("")
                        else:
                            intermed_list.append(phrase_text_b)
                        previous_phrase = phrase_text_b
                        table.append(intermed_list)
        mytable = tabulate.tabulate(table, headers=('Livre', 'Partie', 'Chapitre', 'Latin', 'Castillan'), tablefmt='html')
        as_tree = etree.fromstring(mytable)
        as_tree.set("rules", "all")
        utils.save_tree_to_file(as_tree, "/home/mgl/Téléchargements/table.html")
    
    def alignementMultilingue(self):
        main_file_path, main_file_tree = self.main_file
        for path, tree in self.target_parsed_files.items():
            for chapter in tree.xpath("descendant::tei:div[@type='chapitre']", namespaces=self.tei_ns):
                source_tokens, target_tokens = list(), list()
                target_dict = {}
                source_dict = {}
                chapter_n = chapter.xpath("@n")[0]
                part_n = chapter.xpath("ancestor::tei:div[@type='partie']/@n", namespaces=self.tei_ns)[0]
                book_n = chapter.xpath("ancestor::tei:div[@type='livre']/@n", namespaces=self.tei_ns)[0]
                print(f"Treating {book_n}-{part_n}-{chapter_n}")
                for index, phrase in enumerate(chapter.xpath("descendant::tei:phr", namespaces=self.tei_ns)):
                    ident = utils.generateur_id(6)
                    phrase.set('{http://www.w3.org/XML/1998/namespace}id', ident)
                    target_dict[index] = ident
                    target_tokens.append(' '.join([token.text for token in phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]", namespaces=self.tei_ns)]))

                for index, phrase in enumerate(
                        main_file_tree.xpath(f"descendant::tei:div[@type='livre'][@n='{book_n}']/"
                                             f"descendant::tei:div[@type='partie'][@n='{part_n}']/"
                                             f"descendant::tei:div[@type='chapitre'][@n='{chapter_n}']/"
                                             f"descendant::tei:phr", namespaces=self.tei_ns)):
                    ident = utils.generateur_id(6)
                    phrase.set('{http://www.w3.org/XML/1998/namespace}id', ident)
                    source_dict[index] = ident
                    source_tokens.append(' '.join([token.text for token in phrase.xpath("descendant::node()[self::tei:pc or self::tei:w]", namespaces=self.tei_ns)]))
                assert len(source_dict) == len(source_tokens), f'Error {len(source_dict)} {len(source_tokens)}'
                assert len(target_dict) == len(target_tokens), 'Error'
                aligner = Bertalign(source_tokens, target_tokens)
                aligner.align_sents()
                # aligner.print_sents() 
                tsource = []
                for tuple in aligner.result:
                    source, target = tuple
                    transformed_source = '#' + ' #'.join([source_dict[index] for index in source])
                    transformed_target = '#' + ' #'.join([target_dict[index] for index in target])
                    tsource.append((transformed_source,transformed_target))
                source_target_dict = {source:target for source, target in tsource}
                target_source_dict = {target:source for source, target in tsource}
                
                all_phrases = tree.xpath("descendant::tei:phr", namespaces=self.tei_ns)
                all_ids = tree.xpath("descendant::tei:phr/@xml:id", namespaces=self.tei_ns)
                ids_and_phrases = list(zip(all_ids, all_phrases))
                
                for index, (identifier, phrase) in enumerate(ids_and_phrases):
                    try:
                        match = [id for id in target_source_dict if identifier in id][0]
                        phrase.set('corresp', target_source_dict[match])
                    except IndexError:
                        phrase.set('corresp', 'None')
                        
                    
                
    
                all_phrases = main_file_tree.xpath(f"descendant::tei:div[@type='livre'][@n='{book_n}']/"
                                             f"descendant::tei:div[@type='partie'][@n='{part_n}']/"
                                             f"descendant::tei:div[@type='chapitre'][@n='{chapter_n}']/"
                                             f"descendant::tei:phr", namespaces=self.tei_ns)
                all_ids = main_file_tree.xpath(f"descendant::tei:div[@type='livre'][@n='{book_n}']/"
                                             f"descendant::tei:div[@type='partie'][@n='{part_n}']/"
                                             f"descendant::tei:div[@type='chapitre'][@n='{chapter_n}']/"
                                             f"descendant::tei:phr/@xml:id", namespaces=self.tei_ns)
                ids_and_phrases = list(zip(all_ids, all_phrases))
                for index, (identifier, phrase) in enumerate(ids_and_phrases):
                    try:
                        match = [id for id in source_target_dict if identifier in id][0]
                        phrase.set('corresp', source_target_dict[match])
                    except IndexError:
                        phrase.set('corresp', 'None')
            
                utils.save_tree_to_file(tree, path.replace(".xml", ".final.xml"))
                utils.save_tree_to_file(main_file_tree, main_file_path.replace(".xml", ".final.xml"))

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
    # TODO: transformer en dictionnaire en indiquant clairement qui est le témoin-source
    # files = {"main_file": "/projects/users/mgillele/alignment/bertalign/text+berg/local_data/xml/Rome_W.xml", 
    #          "target_files": ["/projects/users/mgillele/alignment/bertalign/text+berg/local_data/xml/Val_S.citable.xml"]
    #          }
    files = {"main_file": "text+berg/xml/Rome_W.regularized.phrased.xml", 
              "target_files": ["text+berg/xml/Val_S.citable.regularized.phrased.xml"]
              }
    
    arguments = argparse.ArgumentParser()
    arguments.add_argument("-t", "--tokenize", help="Tokenize?", default=True)
    arguments = arguments.parse_args()
    tokenize = arguments.tokenize
    if tokenize == "True":
        tokenize = True
    else:
        tokenize = False
    Aligner = TEIAligner(files, tokenize=tokenize)
    Aligner.create_alignment_table('text+berg/xml/Rome_W.final.xml', 'text+berg/xml/Val_S.citable.final.xml')
    exit(0)
    Aligner.alignementMultilingue()