from bertalign import Bertalign



spanish = "text+berg/latin_castillan/Val_S_1_2_5.txt"
latin =  "text+berg/latin_castillan/Rome_W_1_2_5.txt"

with open(spanish, "r") as spanish_file:
    spanish = spanish_file.read()
    
with open(latin, "r") as latin_file:
    latin = latin_file.read()

aligner = Bertalign(latin, spanish)
aligner.align_sents()
aligner.print_sents()
