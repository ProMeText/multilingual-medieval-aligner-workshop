import unittest


# class AlignmentTests(unittest.TestCase):
#     def __init__(self, align_dict, tokenized_witnesses):
#         self.alignment_dict = align_dict
#         self.tokenized_witnesses = tokenized_witnesses
# 
#     def test_tokenization(self):
#         """
#         On teste si la tokénisation a fonctionné et si une expression régulière ne fait pas tout capoter
#         """
#         self.assertEqual(True, False)
# 
#     def test_tokens_loss(self):
#         """
#         On teste la perte éventuelle de tokens lors du processus
#         """
#         self.assertEqual(True, False)
# 
#     def test_tokens_order(self):
#         """
#         On teste l'inversion éventuelle de tokens lors du processus
#         """
#         self.assertEqual(True, False)
# 
#     def test_tables_consistency(self, align_dict, witnesses):
#         """
#         Cette fonction teste si tous les témoins contiennent bien l'intégralité du texte dans le bon ordre à la fin du processus
#         """
#         for witness in witnesses:
#             print(witness)
#             wit_table = []
#             for alignment_unit in align_dict:
#                 wit_table.extend(int(item) for item in alignment_unit[witness])
#             last_pos = wit_table[-1]
#             ranges = list(range(last_pos + 1))
#             is_equal = wit_table == ranges
#             if is_equal is False:
#                 print("Not right")
#                 print(list(zip(ranges, wit_table)))
#                 print(type(ranges), type(wit_table))
#                 print([(a, b) for a, b in list(zip(ranges, wit_table)) if a != b])
#                 print(align_dict)
#             else:
#                 print("OK")
#         return


# class TokenizationTests(unittest.TestCase):
#     def __init__(self, tokenized_witnesses: list):
#         self.tokenized_witnesses = tokenized_witnesses
# 
#     def test_tokenization(self):
#         """
#         On teste si la tokénisation a fonctionné et si une expression régulière ne fait pas tout capoter
#         """
#         self.assertEqual(True, False)
# 
# 
# if __name__ == '__main__':
#     unittest.main()
