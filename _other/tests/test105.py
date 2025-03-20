list1 = [1, [2, 3, [4, 5]], [6, 7]]

def unpack(itm: list | int | None) -> list | str | None:
    if isinstance(itm, int):
        return hex(itm)
    elif isinstance(itm, list):
        return [unpack(i) for i in itm]
    else:
        return None

print(unpack(list1))
