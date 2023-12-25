import re

# On cherche un alignement au niveau du syntagme et non pas de la phrase. Il faut donc réécrire 
# la fonction de tokénisation pour intégrer les subordonnants, etc ainsi que plus d'éléments de ponctuation, 
# que ce soit en latin ou en castillan.

def split(string:str) -> list:
    print(string)
    # On va utiliser des subordonnant comme séparateurs pour aller au niveau du syntagme
    string = string.replace("\n", " ")
    separator = r"([,;!?.:?¿¶·]| cum |donde| [Qq]ue | ut |por ?que)"
    separated = re.sub(separator, r"||\1", string)
    separated = re.sub(r"\s+", " ", separated)
    splits = re.split("\|\|", separated)
    splits = [split.replace("  ", " ") for split in splits]
    return splits


if __name__ == '__main__':
    spanish = "text+berg/latin_castilian/Val_S_1_2_5.txt"
    latin = "text+berg/latin_castilian/Rome_W_1_2_5.txt"

    with open(spanish, "r") as spanish_file:
        spanish = spanish_file.read()

    with open(latin, "r") as latin_file:
        latin = latin_file.read()

    print(split(spanish))
    print(split(latin))
    
    