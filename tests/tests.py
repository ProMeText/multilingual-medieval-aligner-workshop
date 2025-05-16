import unittest
import main

class Main(unittest.TestCase):
    """
    Cette classe teste l'intégralité de la chaîne
    """
    def test_full_workflow(self):
        virtualArgs = {'input_dir': 'tests/test_data/', 
                       'out_dir': 'result_dir/', 
                       'use_punctuation': True, 
                       'main_wit': 'tests/test_data/es/lanzarote-ii-48.txt', 
                       'prefix': None, 
                       'device': 'cpu'}
        input_dir = virtualArgs['input_dir']
        out_dir = virtualArgs['out_dir']
        main_wit = virtualArgs['main_wit']
        prefix = virtualArgs['prefix']
        device = virtualArgs['device']
        use_punctuation = virtualArgs['use_punctuation']
        
        result = main.run_alignments(out_dir=out_dir, 
                                     input_dir=input_dir, 
                                     main_wit=main_wit, 
                                     prefix=prefix, 
                                     device=device, 
                                     use_punctuation=use_punctuation, 
                                     tokenizer="regexp", 
                                     tok_models=None,
                                     multilingual=True,
                                     corpus_limit=0.08)
        expected_results = {'a': True, 'b': True, 'c': True, 'd': True, 'e': True, 'f': True,  'g':True}
        
        self.assertEqual(result, expected_results)


if __name__ == '__main__':
    unittest.main()