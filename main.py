import numpy as np
import matplotlib.pyplot as plt
from Functions.Extract_Features import signalanalyser
from Functions import loadmatfile
from Functions.Extract_Features import load_samples
from Functions.Extract_Features import load_trn_tst_examples
from Functions.ShowData import display_data
from Functions.ShowData import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def trainclassifier(x, y):
    """Perceptron algorithm to maximize predicted_y = <W,x> + b with parameters weights
    (features, classes) and biases.
    param x: features of the input signal, shape(features,subsignals)
    param y: labels of the every subsignal shape(1, subsignals)
    returns: trained and bias added weight vectors, which will implement
    classification task.
    """
    classes = [0, 1, 2]
    #initialize weight vectors with bias
    w = np.zeros([x.shape[0], 3])
    ones = np.ones([1, 3])
    w = np.insert(w, w.shape[0], ones, axis=0)
    #classes definition matching weight vectors columns
    #initialized at 1000 for longer iteration
    error = 1000
    while error > 0:
        for i in range(len(y)): #every signal with min. length
            argmax = predictclass(w, x[:, i])
            predicted_y = classes[argmax]
            if not (predicted_y == y[i]): #train weight_vectors for each incorrect predicted_y
                error += 1
                for k in range(w.shape[1]): #iterate over the columns for each class, 0, 1, 2
                    w[:,k] = w[:, k] + (classes[k] == trn_Y[i])*np.insert(
                        trn_X[:, i], 0, 1, axis=0)-\
                        (k == argmax)*np.insert(trn_X[:, i], 0, 1, axis=0)
            else:
                error -= 1
    return w


def predictclass(w, x):
    """Returns max argument for scalar product of input weight vector and
    (bias added) input feature vector of feature, which is the predicted
    class"""
    if x.ndim > 1:
        a = []
        for i in range(x.shape[1]):
            predicted_y = np.argmax(np.dot(w.T, np.insert(x[:, i], 0, 1, axis=0)))
            a.append(predicted_y)
        return max(set(a), key=a.count)
    else:
        predicted_y = np.argmax(np.dot(w.T, np.insert(x, 0, 1, axis=0)))
        return predicted_y


def tstclassifier(w, tst_X, tst_Y):
    """test trained weights with test data, will return the error rate
     on dataset X, Y."""

    classes = [0, 1, 2]
    output = ["Healthy", "Inner race fault", "Outer race fault"]
    error = 0
    counter = 0
    for i in range(len(tst_Y)):
        counter += 1
        predicted_y = predictclass(w, tst_X[:, i])
        print(output[predicted_y])
        if not predicted_y == tst_Y[i]:
            error += 1
    error_rate = error/counter*100
    print(f"{error} error in {counter} subsignal")
    print(f"error rate is %{error_rate} ")
    return error_rate


def confusion_m(w, x, y):
    """Another accuracy representation with confusion matrix"""
    classes = [0, 1, 2]
    actual = []
    predicted = []
    output = ["Healthy", "Inner race fault", "Outer race fault"]
    for i in range(len(y)):
        predicted_y = predictclass(w, x[:, i])
        predicted.append(output[predicted_y])
        actual.append(output[y[i]])

    cnf_matrix = confusion_matrix(actual, predicted, labels=output)
    return cnf_matrix

def tstsinglefile(w, filename):
    """This function is to analyze a signal with its data in .mat file
    in the current directory.
    params: filename of the signal .mat file
    returns: healthy, inner race fault or outer race fault based on
    the features of the signal."""

    data_dict = loadmatfile.load_single_example(filename)
    signal = signalanalyser(data_dict)
    output = ["Healthy", "Inner race fault", "Outer race fault"]
    X_, labels = signal.extract_features()

    return output[predictclass(w, X_)]


if __name__ == "__main__":
    all_datas = loadmatfile.load_datas("Bearing Data")
    load_samples(all_datas)
    normal_x = np.load("normal_X.npy")
    inner_x = np.load("inner_X.npy")
    outer_x = np.load("outer_X.npy")
    display_data(normal_x, inner_x, outer_x)
    trn_X, trn_Y, tst_X, tst_Y = load_trn_tst_examples(normal_x, inner_x, outer_x)
    w = trainclassifier(trn_X, trn_Y)
    c_m = confusion_m(w, tst_X, tst_Y)
    plt.figure()
    plot_confusion_matrix(c_m, classes=["Healthy", "Inner race fault", "Outer race fault"])
    plt.show()
    #np.save("trained_weight.npy", w)
    #w = np.load("trained_weight.npy")
    tstclassifier(w, tst_X, tst_Y)
    #print(tstsinglefile(w, "OuterRaceFault_5.mat"))
