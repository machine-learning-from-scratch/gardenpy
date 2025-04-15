dict1 = {
    'numbers': [1, 2, 3, 4, 5],
    'letters': ['a', 'b', 'c', 'd', 'e']
}

list1 = ['v', 'w', 'x', 'y', 'z']

for list_itm, number, letter in zip(list1[::-1], *[value[::-1] for value in dict1.values()]):
    print(list_itm, number, letter)

for number, letter in zip(*[value[::-1] for value in dict1.values()]):
    print(number, letter)
