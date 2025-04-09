from models.dataloader import MNISTFNNDataLoader

dataloader = MNISTFNNDataLoader(
    root=...,
    values_path='values.csv',
    labels_path='labels.csv',
    save_to_memory=True
)

for i in range(5):
    values, labels = next(iter(dataloader))
    print(values)
    print('----------')
    print(labels)
    print('----------')
