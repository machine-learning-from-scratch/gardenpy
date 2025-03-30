relation_list = [[1, 2, 3, 4, 5], [1, 4, 2, 5], [1, 2, 6, 5]]

final_relation = [1] + [rlt[1:-1] for rlt in relation_list] + [5]

print(final_relation)
