# loading processed_data for testing
from PIL import Image
from _other.deprecated.z_fnn_old import NeuralNetwork as nnet
import numpy as np
from tqdm import tqdm
import pandas as pd
import ast
import random
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

keras_weights = []
keras_biases = []
with open(f"../../../assets/saved/weights_keras.txt", "r") as f:
    for line in f:
        keras_weights.append(np.array(ast.literal_eval(line)))
with open(f"../../../assets/saved/biases_keras.txt", "r") as f:
    for line in f:
        keras_biases.append(np.array(ast.literal_eval(line)))
img = Image.open(f"../../assets/saved/user_number.jpeg")
gray_img = img.convert("L")
test_input = np.array(list(gray_img.getdata())) / 255

# load MNIST processed_data
df_values = np.array(pd.read_csv(f"../../../assets/processed_data/FNN_mnist/FNN_mnist_values.csv")).tolist()
for i in tqdm(range(len(df_values)), ncols=150, desc="Reformatting Data Values"):
    df_values[i] = np.array([df_values[i]])
df_labels = np.array(pd.read_csv(f"../../../assets/processed_data/FNN_mnist/FNN_mnist_labels.csv")).tolist()
for i in tqdm(range(len(df_labels)), ncols=150, desc="Reformatting Data Labels"):
    df_labels[i] = np.array([df_labels[i]])

# df_values = pd.read_csv(f'assets/processed_data/values.csv')
# df_labels = pd.read_csv(f'assets/processed_data/labels.csv')

def test_train_split(data, test_size):
    random.shuffle(data)
    test = data[0:round(len(data) * test_size)]
    train = data[round(len(data) * test_size):]
    return train, test


def softmax(values):
    output = np.exp(values) / np.sum(np.exp(values))
    return output


# split training and testing processed_data
train, test = test_train_split(list(zip(df_values, df_labels)), test_size=0.3)
# unzip training and testing processed_data
X, Y = zip(*train)
X_test, Y_test = zip(*test)
X, Y = list(X), list(Y)
# reformat training and testing processed_data
X_test, Y_test = list(X_test), list(Y_test)

# class loading
neural_net = nnet(layer_sizes=[784, 16, 16, 10], activation="leaky relu")
neural_net.configure_reporting(loss_reporting=True)
neural_net.validation(valid_x=X_test, valid_y=Y_test)
neural_net.fit(X, Y, max_iter=1000)
# class results
results = neural_net.get_results()
# print(results['mean loss'], results['mean validation loss'], results['accuracy'], results['validation accuracy'])
# print(results['validation outcomes']['loss'])
# print(results['validation confusion matrix'])

# confusion matrices
disp = ConfusionMatrixDisplay(
        confusion_matrix=results['training confusion matrix'],
    )
disp.plot(cmap='Blues')
disp.ax_.set_title('training confusion matrix')
plt.show()
disp = ConfusionMatrixDisplay(
        confusion_matrix=results['validation confusion matrix'],
    )
disp.plot(cmap='Blues')
disp.ax_.set_title('validation confusion matrix')
plt.show()

# plotting
plt.plot(results['logged points'], results['train losses'], color="blue", alpha=0.5, label="Train")
plt.plot(results['logged points'], results['validation losses'], color="red", alpha=0.5, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss v.s. Epoch")
plt.legend(loc="lower right")
plt.show()


df = {
    'smax_pred': [],
    'smax_cor_pred': [],
    'pred': [],
    'cor_pred': [],
    'cor': [],
    'label': []
}
for i in range(len(df_values)):
    predicted = neural_net.forward(df_values[i])[-1][0]
    # print(predicted)
    smax = softmax(predicted)
    # print(smax)
    df['smax_pred'].append(np.max(smax) * 100)
    df['smax_cor_pred'].append(smax[np.nanargmax(df_labels[i])] * 100)
    df['pred'].append(np.max(predicted))
    df['cor_pred'].append(predicted[np.nanargmax(df_labels[i])])
    df['cor'].append(np.nanargmax(predicted) == np.nanargmax(df_labels[i]))
    df['label'].append(np.nanargmax(df_labels[i]))
df = pd.DataFrame(df)
print(df)
sns.violinplot(data=df, x='label', y='smax_pred', hue='cor', split=True, gap=.1, inner='quart', density_norm='count', palette='Greens_d')
plt.show()
sns.violinplot(data=df, x='label', y='smax_cor_pred', hue='cor', split=True, gap=.1, inner='quart', density_norm='count')
plt.show()
sns.violinplot(data=df, x='label', y='pred', hue='cor', split=True, gap=.1, inner='quart', density_norm='count')
plt.show()
sns.violinplot(data=df, x='label', y='cor_pred', hue='cor', split=True, gap=.1, inner='quart', density_norm='count')
plt.show()

neural_net.print_credits()
