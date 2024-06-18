import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import datasets

plt.rcParams['figure.figsize'] = 15, 10

def encode_label(j):
    # 5 -> [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784,1)) for x in data]
    #print('features\n', len(features[0]))
    labels = [encode_label(y[1]) for y in data]
    #print('labels\n', len(labels[0]))
    return zip(features, labels)

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def predict(x, W, b):
    return sigmoid(np.dot(W,x) + b)

def get_predictions(data, W, b):
    predicted_labels = []
    actual_labels = []

    for x, y in data:
        predicted_probabilities = [predict(x, W[i], b) for i in range(10)]
        predicted_label = np.argmax(predicted_probabilities)
        predicted_labels.append(predicted_label)
        actual_label = np.argmax(y)
        actual_labels.append(actual_label)

    return predicted_labels, actual_labels



transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5), (0.5))])
# Downloading the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)


test1 = train_dataset[0]
# (Img, label)
test2 = [test1]
reshape = shape_data(test2)
list(reshape)

train = shape_data(train_dataset)
test = shape_data(test_dataset)
train = list(train)
test = list(test)

avg_zero = average_digit(train, 0)
avg_one = average_digit(train, 1)
avg_two = average_digit(train, 2)
avg_three = average_digit(train, 3)
avg_four = average_digit(train, 4)
avg_five = average_digit(train, 5)
avg_six = average_digit(train, 6)
avg_seven = average_digit(train, 7)
avg_eight = average_digit(train, 8)
avg_nine = average_digit(train, 9)

W_zero = np.transpose(avg_zero)
W_one = np.transpose(avg_one)
W_two = np.transpose(avg_two)
W_three = np.transpose(avg_three)
W_four = np.transpose(avg_four)
W_five= np.transpose(avg_five)
W_six = np.transpose(avg_six)
W_seven = np.transpose(avg_seven)
W_eight = np.transpose(avg_eight)
W_nine = np.transpose(avg_nine)

W = [W_zero, W_one, W_two, W_three, W_four, W_five, W_six, W_seven, W_eight, W_nine]

b = -72

predicted_labels, actual_labels = get_predictions(test, W, b)
accuracy = accuracy_score(actual_labels, predicted_labels)
print("Accuracy:", accuracy)

labels = []
embedded_vectors = []
ind = [0]*10
features = []
for image, label in train:
    lb = np.argmax(label)
    if (all(ind[i] >= 30 for i in range(10))):
        break
    if (ind[lb]) < 30:
        features.append(image)
        ind[lb]+=1
        labels.append(lb)
features = np.array(features)
features = np.squeeze(features)


tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(features)
plt.scatter(tsne[:, 0], tsne[:, 1], s= 10, c=labels, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Samples from training data', fontsize=24);
plt.show()


embedded_vectors = []

for i in range(len(features)):
    x = features[i]
    y = labels[i]
    embedded_vector = [predict(x, W[j], b) for j in range(10)]
    embedded_vectors.append(embedded_vector)


embedded_vectors = np.array(embedded_vectors)
embedded_vectors = np.squeeze(embedded_vectors)

# Применение t-SNE для визуализации эмбеддингов
tsne_embedded = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=300).fit_transform(embedded_vectors)

# Визуализация эмбеддингов
plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], s=10, c=labels, cmap='Spectral')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('Results on Training data', fontsize=24)
plt.show()