import re

# On cherche un alignement au niveau du syntagme et non pas de la phrase. Il faut donc réécrire 
# la fonction de tokénisation pour intégrer les subordonnants, etc ainsi que plus d'éléments de ponctuation, 
# que ce soit en latin ou en castillan.

def split(string:str) -> list:
    # On va utiliser des subordonnant comme séparateurs pour aller au niveau du syntagme
    separator = r"[,;!?.:?¿]|( cum |donde| [Qq]ue | ut |·|¶)"
    splits = re.split(separator, string)
    splits = [split for split in splits if split]
    cleaned_list = []
    for index, split in enumerate(splits):
        if len(split.split()) == 1:
            try:
                cleaned_list.append(f"{split} {splits[index + 1]}")
            except IndexError:
                cleaned_list.append(split)
        elif len(splits[index - 1].split()) == 1:
            pass
        else:
            cleaned_list.append(split)
    splits = [split.replace("  ", " ") for split in cleaned_list]
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
    
    