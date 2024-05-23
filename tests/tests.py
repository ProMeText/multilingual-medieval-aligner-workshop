import unittest
import main

class Main(unittest.TestCase):
    def test_full_workflow(self):
        virtualArgs = {'input_dir': 'test_data/', 
                       'out_dir': 'result_dir/', 
                       'use_punctuation': True, 
                       'main_wit': 'test_data/castillan/lanzarote-ii-48.txt', 
                       'prefix': None, 
                       'device': 'cuda:0'}
        input_dir = virtualArgs['input_dir']
        out_dir = virtualArgs['out_dir']
        main_wit = virtualArgs['main_wit']
        prefix = virtualArgs['prefix']
        device = virtualArgs['device']
        use_punctuation = virtualArgs['use_punctuation']
        
        result = main.run_alignments(out_dir, input_dir, main_wit, prefix, device, use_punctuation, corpus_size=100)
        expected_results = {'a': True, 'b': True, 'c': True, 'd': True, 'e': True, 'f': True,  'g':True}
        
        self.assertEqual(result, expected_results)


if __name__ == '__main__':
    unittest.main()