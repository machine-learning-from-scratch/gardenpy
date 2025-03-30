# decorator showcase test

def print_arguments(func):
    def wrapper(*args, **kwargs):
        print(args, kwargs)
        func(*args, **kwargs)
    return wrapper


def add_nums(itm1, itm2):
    print(itm1 + itm2)


@print_arguments
def add_nums_decorator(itm1, itm2):
    print(itm1 + itm2)


add_nums(5, 6)

add_nums_decorator(5, 6)
