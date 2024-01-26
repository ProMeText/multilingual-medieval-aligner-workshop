from bertalign import Bertalign
import sys

spanish = sys.argv[1]
latin = sys.argv[2]

with open(spanish, "r") as spanish_file:
    spanish = spanish_file.read().split("\n")
    
with open(latin, "r") as latin_file:
    latin = latin_file.read().split("\n")

aligner = Bertalign(latin, spanish)
aligner.align_sents()
print(aligner.result)
aligner.print_sents()
