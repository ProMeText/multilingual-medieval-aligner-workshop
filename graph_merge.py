import networkx as networkx

def desambiguise(object, labels):
    """
    On ajoute à chaque noeud le document d'origine:
    ((0, 0), 
    (1, 1), 
    (2, 2), 
    (3, 3), etc] 
    -> 
    [(('0_a',), ('0_b',)), 
    (('1_a',), ('1_b',)),
    etc]
    """
    as_unique_nodes = []
    for alignment_unit in object:
        A, B = alignment_unit
        if isinstance(A, int): A = (A,)
        if isinstance(B, int): B = (B,)
        ranges_A = []
        ranges_B = []
        for element in A:
            ranges_A.append(f"{str(element)}_{labels[0]}")
        for element in B:
            ranges_B.append(f"{str(element)}_{labels[1]}")
        as_unique_nodes.append((tuple(ranges_A), tuple(ranges_B)))
    return as_unique_nodes

def deconnect(object):
    """
    On passe de groupes à des paires de positions
    [(('5_a', '6_a', '7_a'), ('5_b', '6_b'))]
    -> [ ('5_a', '5_b'), ('5_a', '6_b'), ('6_a', '5_b'), ('6_a', '6_b'), ('7_a', '5_b'), ('7_a', '6_b')]
    """
    
    final_list = []
    for alignment_unit in object:
        A, B = alignment_unit
        for item_A in A:
            for item_B in B:
                final_list.append((item_A, item_B))
    return tuple(final_list)

def merge_alignment_table(alignment_dict:dict) -> list:
    """
    Cette fonction va permettre de fusionner les items d'alignement à l'aide d'un passage par le graphe.
    Il en ressort une liste de dictionnaires, chaque item consistant en une unité d'alignement.
    """    
    # On désambiguise les noeuds
    G = networkx.petersen_graph()
    # On modifie la structure pour avoir des noeuds connectés 2 à 2 et des tuples
    string = "abcdefghijk"
    for index, value in alignment_dict.items():
        structured_a = deconnect(desambiguise(value, (string[0], string[index + 1])))
        print(structured_a)
        # Résultat de la forme: (('0_a', '0_b'), ('1_a', '1_b'), ('2_a', '2_b'), ('3_a', '3_b'), etc.)
        G.add_edges_from(structured_a)
    connected_nodes = []
    # On prend chaque noeud et on en ressort les noeuds connectés
    for node in G:
        # https://stackoverflow.com/a/33089602
        connected_components = list(networkx.node_connected_component(G, node))
        connected_components.sort()
        connected_nodes.append(tuple(connected_components))

    # On supprime les noeuds redondants
    connected_nodes = list(set(connected_nodes))
    # Liste de la forme: [('27_a', '27_c', '28_a', '28_c', '29_c', '30_c', '33_b', '34_b'), ('19_c', '20_a', '20_c', '22_b'), etc.]
    
    # Ici c'est un bug: il y a des noeuds qui sont des entiers, je ne sais pas pourquoi. On les supprime.
    connected_nodes = [group for group in connected_nodes if not isinstance(group[0], int)]
    
    # On trie les noeuds par position
    connected_nodes.sort(key=lambda x: int(x[0].split('_')[0]))
    nodes_as_dict = []
    for connection in connected_nodes:
        wit_dictionnary = {}
        for document in string[:len(alignment_dict) + 1]:
            wit_dictionnary[document] = [node.replace(f'_{document}', '') for node in connection if document in node]
        nodes_as_dict.append(wit_dictionnary)
    return nodes_as_dict





if __name__ == '__main__':
    result_b = (((0), (0)), ((1, 2, 3), (1, 2)), ((4, 5), (3)), ((6, 7), (4)), ((8), (5)),
                ((9, 10, 11), (6)), ((12), (7, 8)), ((13), (9, 10)), ((14), (11)),
                ((15), (12)), ((16, 17), (13, 14, 15)), ((18), (16)), ((19), (17)),
                ((), (18)), ((20), (19, 20)), ((21), (21, 22)), ((22, 23, 24), (23)),
                ((25, 26), (24, 25, 26)), ((27), (27)), ((28), (28, 29, 30)), ((29), (31)),
                ((30, 31, 32), (32)), ((33), (33)), ((34, 35), (34)), ((36, 37), (35)),
                ((38), (36)), ((39), (37, 38, 39)))

    result_a = (((0), (0)), ((1), (1)), ((2), (2)), ((3), (3)), ((4), (4)), ((5, 6, 7), (5, 6)),
                ((8), (7, 8, 9, 10)), ((9), (11, 12)), ((10, 11), (13)), ((12), ()), ((13), (14)),
                ((14), (15, 16)), ((15), (17, 18)), ((16), (19)), ((17, 18, 19), (20, 21)),
                ((20), (22)), ((21), (23, 24)), ((22), (25)), ((23), (26)), ((24), (27)),
                ((25), (28, 29, 30)), ((26), (31, 32)), ((27, 28), (33, 34)), ((29), (35)),
                ((30, 31), (36, 37)), ((32), (38)), ((33), (39)), ((34), ()), ((35), ()),
                ((36), ()), ((37), ()), ((38), ()), ((39), ()))
    result = merge_alignment_table(result_a, result_b)
    print(result)
    