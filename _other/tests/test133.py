import random

while True:
    num = int(input("Num: "))
    nums = [random.randint(31, 216) for _ in range(num)]
    for itm in nums:
        if 15 < itm:
            print(f"#{hex(itm)[2:] * 3}", end=" ")
        else:
            print(f"#{f'0{hex(itm)[2:]}' * 3}", end=" ")
    print("")
