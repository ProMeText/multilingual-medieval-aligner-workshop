import re


def split(string:str) -> list:
    separator = r"[,;!?:?¿]"
    splits = re.split(separator, string)
    return splits