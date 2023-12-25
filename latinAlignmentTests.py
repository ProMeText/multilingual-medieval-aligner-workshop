from bertalign import Bertalign



spanish = "text+berg/latin_castilian/Val_S_1_2_5.txt"
spanish_2 = "text+berg/latin_castilian/Sev_Z_1_2_5.txt"
latin =  "text+berg/latin_castilian/Rome_W_1_2_5.txt"

with open(spanish, "r") as spanish_file:
    spanish = spanish_file.read()
    
with open(spanish_2, "r") as spanish_file_2:
    spanish_2 = spanish_file_2.read()
    
with open(latin, "r") as latin_file:
    latin = latin_file.read()

aligner = Bertalign(spanish_2, spanish)
aligner.align_sents()
aligner.print_sents()
