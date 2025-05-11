from gardenpy import utils
import os

loader = utils.dataloaders.DataLoaderCSV(os.getcwd(), "labels.csv", "values.csv", 2)
loader.load()
print(loader._values.get_internals())
loader.__next__()
print(loader._values.get_internals())

loader.shuffle()
print(loader.order)