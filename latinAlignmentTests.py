from bertalign import Bertalign
import sys

spanish = sys.argv[1]
latin = sys.argv[2]

with open(spanish, "r") as spanish_file:
    spanish = spanish_file.read()
    
with open(spanish_2, "r") as spanish_file_2:
    spanish_2 = spanish_file_2.read()
    
with open(latin, "r") as latin_file:
    latin = latin_file.read()

aligner = Bertalign(latin, spanish)
aligner.align_sents()
aligner.print_sents()
