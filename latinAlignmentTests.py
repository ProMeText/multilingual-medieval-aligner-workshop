from bertalign import Bertalign



spanish = "text+berg/test_lat-spo/sev_z.txt"
latin =  "text+berg/test_lat-spo/rome_w.txt"

with open(spanish, "r") as spanish_file:
    spanish = spanish_file.readlines()
with open(latin, "r") as latin_file:
    latin_file = spanish_file.readlines()

aligner = Bertalign(latin, spanish)
aligner.align_sents()
aligner.print_sents()
