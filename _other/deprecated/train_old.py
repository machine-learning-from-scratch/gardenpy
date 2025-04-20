# todo: deprecate

"""
Feedforward Neural Network training for MNIST
"""

import os

from gardenpy.metrics import (
    visualization as vis
)
from gardenpy.utils import (
    helpers as hf
)
from models import dataloader as du

from _other.deprecated.z_fnn_old import (
    FNN
)

import demo.mnist.dnn_mnist.processed_data.process as p

# locations
root = os.path.join(os.path.dirname(__file__))
saved_location = os.path.join(root, 'model')
data_location = os.path.join(root, 'data', 'processed_data')

# hyper-parameters
# note: some hyper-parameters might have strange names as FNN gets restructured
layer_sizes = [784, 16, 16, 10]  # network layer sizes
activation = 'leaky-relu'  # network activation function
solver = 'mini-batch'  # type of data batching (solver is not the correct name and should be 'batching', will be changed later)
cost_function = 'mse'  # the loss/cost function (currently, only mse works, and l1 might work but is not very good)
optimizing_method = 'gradient descent'  # not implemented yet, ignore for now (don't enter as an argument)
batch_size = 49  # batch size for mini-batching
learning_rate = 0.001  # learning rate
max_iter = 10000  # the maximum amount of batches trained on
alpha = 0.001  # weight decay (l2 regularization) term (incorrect name, 'lambda_d' is the correct term, will change later)
trim_data = False  # select only a portion of the dataset (optional)
trim_frac = 0.01  # percent of dataset taken if trimming (optional)
set_validation = True  # set a validation set to compare results to (optional)
val_frac = 0.3  # the fraction of data that goes to validation if there is validation set (optional)
loss_reporting = True  # log the loss as the model trains
eval_batch_size = 70  # batch size of the data taken for loss calculations
eval_interval = max(1, round(max_iter * 0.01))  # how often the loss of the model is logged
load_parameters = False  # load an existing model
weights_name = 'weights.txt'  # the location of weights of an existing model
biases_name = 'biases.txt'  # the location of biases of an existing model
save_data = True  # save data to a file if the file didn't exist previously and needed to be generated
values_name = 'values.csv'  # the location of the inputs of the data
labels_name = 'labels.csv'  # the location of the labels of the data

conf_mat_normal = True  # the type of normalization done on the confusion matrix for results

# print gardenpy credits
hf.print_credits()

# import assets
weights = None
biases = None
if not (os.path.isfile(os.path.join(data_location, values_name)) and os.path.isfile(os.path.join(data_location, labels_name))):
    values_file = os.path.join(data_location, values_name)
    labels_file = os.path.join(data_location, labels_name)
    keras_values, keras_labels = p.process_data(save=save_data, values_file=values_file, labels_file=labels_file)
else:
    keras_values = du.format_data(os.path.join(data_location, values_name))
    keras_labels = du.format_data(os.path.join(data_location, labels_name))
if load_parameters:
    if os.path.exists(os.path.join(saved_location, weights_name)):
        weights = du.format_parameters(os.path.join(saved_location, weights_name))
    else:
        raise FileNotFoundError(f'{os.path.join(saved_location, weights_name)} does not exist')
    if os.path.exists(os.path.join(saved_location, biases_name)):
        biases = du.format_parameters(os.path.join(saved_location, biases_name))
    else:
        raise FileNotFoundError(f'{os.path.join(saved_location, biases_name)} does not exist')

# form network
keras_network = FNN(layer_sizes, weights=weights, biases=biases, activation=activation)
# configure loss logging
keras_network.configure_reporting(loss_reporting=loss_reporting, eval_batch_size=eval_batch_size, eval_interval=eval_interval)

# trim processed_data
if trim_data:
    keras_values = du.trim(keras_values, trim_frac=trim_frac)
    keras_labels = du.trim(keras_labels, trim_frac=trim_frac)

# set validation
if set_validation:
    x, y, val_x, val_y = du.test_val(keras_values, keras_labels, val_frac=val_frac)
    keras_network.validation(valid_x=val_x, valid_y=val_y)
else:
    x, y = keras_values, keras_labels

# train model
keras_network.fit(
    x=x,
    y=y,
    solver=solver,
    cost_function=cost_function,
    optimizing_method=optimizing_method,
    batch_size=batch_size,
    learning_rate=learning_rate,
    max_iter=max_iter,
    alpha=alpha,
)

# get model results
keras_results = keras_network.get_results(cm_norm=conf_mat_normal)
# Mtr.print_color(keras_results)
vis.print_results(keras_results)
# graph results
vis.cm_disp((keras_results['training confusion matrix']), title='Training Results', normalized=conf_mat_normal)
vis.cm_disp((keras_results['validation confusion matrix']), title='Validation Results', normalized=conf_mat_normal)
vis.loss_graph(keras_results['logged losses'])
vis.reg_loss_graph(keras_results['logged losses'])
vis.violin_plot(keras_results['training outcomes'], title='Training Violin Plot')
if set_validation:
    vis.violin_plot(keras_results['validation outcomes'], title='Validation Violin Plot')

# save model
save_parameters = hf.input_color("input 's' to save parameters: ")
if save_parameters.lower() == 's':
    du.save_parameters(os.path.join(saved_location, weights_name), keras_results['weights'])
    du.save_parameters(os.path.join(saved_location, biases_name), keras_results['biases'])
else:
    hf.print_color('parameters not saved')
