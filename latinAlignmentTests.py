from bertalign import Bertalign
import sys

spanish = sys.argv[1]
latin = sys.argv[2]

with open(spanish, "r") as spanish_file:
    spanish = spanish_file.read()
    
with open(latin, "r") as latin_file:
    latin = latin_file.read()

aligner = Bertalign(latin, spanish)
aligner.align_sents()
print(aligner.result)
exit(0)
aligner.print_sents()
