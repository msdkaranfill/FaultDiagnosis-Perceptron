import matplotlib.pyplot as plt
import numpy as np

def display_data(normal, inner, outer):
    feature_names = ['MEAN','STD','RMS','SK','KUR','SF','CF',
                    'IF','MF','PEAK','P2P', 'TLF', 'THKT', 'BPFO', 'BPFI']

    features = np.hstack([normal, inner])
    features = np.hstack([features, outer])

    ix_normal = np.arange(0, normal.shape[1]).T
    ix_outer = np.arange(normal.shape[1], normal.shape[1] + outer.shape[1]).T
    ix_inner = np.arange(normal.shape[1] + outer.shape[1], normal.shape[1] +
                         outer.shape[1] + inner.shape[1]).T

    plt.rc('font', size=14)
    plt.figure(1, figsize=(12, 6))
    for ix in range(len(feature_names)):
        plt.scatter(ix * np.ones(ix_normal.shape[0]), features[ix,
                        ix_normal], c='b', marker='o', s=50, alpha=0.5)
        print(features[ix,ix_normal])
        plt.scatter(ix * np.ones(ix_outer.shape[0]), features[ix,
                            ix_outer], c='r', marker='x', s=50, alpha=0.5)
        plt.scatter(ix * np.ones(ix_inner.shape[0]), features[ix,
                            ix_inner], c='g', marker='d', s=50, alpha=0.5)

    plt.xticks(range(len(feature_names)), feature_names)
    plt.xlim(-1, len(feature_names))
    plt.legend(['Healthy', 'Outer', 'Inner'])
    plt.title('Features of the bearing data')
    plt.ylabel('Normalized value')
    plt.grid()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

