r"""
Dense Neural Network training on MNIST.
"""

import os

from gardenpy.models import DNN
from gardenpy.utils import DataLoaderCSV, ansi
from gardenpy.utils.helpers import print_contributors

print()
print_contributors()

dataset = DataLoaderCSV(
    labels='labels',
    values='values',
    root=os.path.join(os.path.dirname(__file__), 'processed_data'),
    valid=0.2,
    batching=128,
    trim=False,
    shuffle=True,
    save_memory=True,
    status_bars=True
)

model = DNN(status_bars=True)
model.configure(
    hidden_layers=[512, 256, 128],
    thetas={
        'weights': [
            {'algorithm': 'xavier', 'mu': 0.0, 'sigma': 1.0},
            {'algorithm': 'xavier', 'mu': 0.0, 'sigma': 1.0},
            {'algorithm': 'xavier', 'mu': 0.0, 'sigma': 1.0},
            {'algorithm': 'xavier', 'mu': 0.0, 'sigma': 1.0}
        ],
        'biases': [
            {'algorithm': 'uniform', 'value': 0.0},
            {'algorithm': 'uniform', 'value': 0.0},
            {'algorithm': 'uniform', 'value': 0.0},
            {'algorithm': 'uniform', 'value': 0.0}
        ]
    },
    activations=[
        {'algorithm': 'relu'},
        {'algorithm': 'relu'},
        {'algorithm': 'relu'},
        {'algorithm': 'softmax'}
    ]
)
model.hyperparameters(
    loss={'algorithm': 'centropy'},
    optimizer={
        'algorithm': 'adam',
        'alpha': 1e-3,
        'lambda_d': 0,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'ams': False
    }
)
model.fit(
    data=dataset,
    parameters={
        'epochs': 50,
        'log_rate': 5
    }
)
weights, biases = model.get_thetas()
save_parameters = input(f"Input '{ansi['white']}{ansi['italic']}save{ansi['reset']}' to save parameters: ")
if save_parameters.lower() == 'save':
    saver = SaveDNN(status_bars=True)
    saver.save(location=...)
else:
    print(f"{ansi['bright_black']}{ansi['italic']}Parameters not saved.{ansi['reset']}")

# evals = EvaluationDNN(status_bars=True)
# evals.evaluate(
#     stats={
#                 'log': {'on': True}
#     }
# )
# evals.graphs(
#     graphs={
#         'boxplot': {'on': True},
#         'heatmap': {'on': True, 'normalized': True, 'validation': True},
#         'lgraph': {'on': True, 'accuracy': True, 'loss': True, 'validation': True},
#         'rgraph': {'on': True, 'accuracy': True, 'loss': True, 'validation': True}
#     }
# )
# print()
