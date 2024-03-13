import numpy as np
from Functions.TimeFeature import timefeature
from Functions.AR_Filter import ar_filter
from Functions.FreqFeatures import fault_amp


#analyze the signal and extract features
class signalanalyser():
    """Class to extract features of given dictionary type data, returns feature matrices."""
    def __init__(self, data_dict):
        self.fs = data_dict["sr"]
        self.fr = data_dict["rate"]
        self.n_samples = int(len(data_dict["gs"])/(self.fs/2)) #corresponding to min. length
        self.datas = self.split_data(data_dict["gs"])
        self.label = data_dict["label"]
        self.bff = np.array([3.245, 4.7550]) * self.fr   #[BPFO BPFI]

        assert self.n_samples, len(self.datas)

    def split_data(self, data):
        """This function will split the input signal with minimum length
        and will return a list of signal arrays."""

        datas = [1e-10] * self.n_samples  #initialize
        for i in range(self.n_samples):
            minl = int(i * (self.fs/2))
            datas[i] = np.asarray(data[minl:int(minl+(self.fs/2))])

        return datas

    def extract_features(self):
        """All features for the input signal are extracted in this function.
        returns: features: X and labels: Y"""
        bw = 1017			         #bandwidth, check !! check back.
        pass
        X_ = np.zeros([15, self.n_samples]) #shape[0]: features, shape[1]: subsignal datas
        Y_ = []
        for i, data in enumerate(self.datas):
            print(f"{i} numbered sub signal is being analyzed for a {self.label} signal")
            #each input signal with 0.5 seconds
            t_feats, f_names = timefeature(data)
            #xr = ar_filter(data, 700)[0] 			#residual signal from AR filter
            xr = data  #uncomment this line and comment ar_filter to see the results
            f_feats, ff_names = fault_amp(xr, self.fs, self.bff, bw)
            features = np.hstack([t_feats, f_feats])
            X_[:, i] = (features - np.mean(features, axis=0))/np.std(features, ddof=2, axis=0)
            Y_.append(self.label) 	#there will be a label for each sub signal

        return X_, Y_




def load_samples(all_datas):
    """This function will take all datas from folder which has the data and will return a
    npy.file in the directory, which has feature matrices for every file in the directory
     named with label names, baseline, OuterRaceFault and InnerRaceFault feature values.
    returns: stored np arrays of different classes' features.
    """

    ns = len(all_datas)  #number of samples
    #writefile = {"trn_X": []*ns, "trn_Y": []*ns, "tst_X": []*ns, "tst_Y": []*ns}
    normal = np.zeros([15, 36])
    outer = np.zeros([15, 30])
    inner = np.zeros([15, 18])
    n_i = 0
    i_i = 0
    o_i = 0
    for data_dict in all_datas:
        print(f"A {data_dict['label']} named file is being analysed... ")
        signal = signalanalyser(data_dict)
        X_, labels = signal.extract_features()
        for j in range(len(labels)):
            #every subsignal in one signal
            if labels[j] == "baseline":
                normal[:,n_i] = X_[:,j]
                n_i += 1
            if labels[j] == "OuterRaceFault":
                outer[:,o_i] = X_[:,j]
                o_i += 1
            if labels[j] == "InnerRaceFault":
                inner[:,i_i] = X_[:, j]
                i_i += 1

    np.save("normal_X.npy", normal)
    np.save("inner_X.npy", inner)
    np.save("outer_X.npy", outer)

    return 1


def load_trn_tst_examples(normal, inner, outer):
    """This function takes feature matrices with each class label and returns
    training and test dataset with features, X and labels, Y
    :params, feature matrices of each class
    returns: train_X, train_Y, test_X and test_Y"""
    trn_Y, tst_Y = [], []
    samples = normal.shape[1] + inner.shape[1] + outer.shape[1]
    CUT_RATIO = 5 / 6
    trn_X = np.zeros([normal.shape[0], int(samples * CUT_RATIO)+1])
    tst_X = np.zeros([normal.shape[0], int(samples * (1-CUT_RATIO))+1])
    tri = 0 #training set indices
    tsi = 0

    for i in range(normal.shape[1]):
        if i < int(normal.shape[1] * CUT_RATIO):
            trn_X[:,tri] = normal[:,i]
            trn_Y.append(0)
            tri += 1
        else:
            tst_X[:,tsi]=normal[:,i]
            tst_Y.append(0)  #0: baseline, #1: Inner, #2 Outer
            tsi += 1
    for i in range(inner.shape[1]):
        if i < int(inner.shape[1] * CUT_RATIO):
            trn_X[:, tri] = inner[:, i]
            trn_Y.append(1)
            tri += 1
        else:
            tst_X[:, tsi] = inner[:, i]
            tst_Y.append(1)
            tsi += 1
    for i in range(outer.shape[1]):
        if i < int(outer.shape[1] * CUT_RATIO):
            trn_X[:, tri] = outer[:, i]
            trn_Y.append(2)
            tri += 1
        else:
            tst_X[:, tsi] = outer[:, i]
            tst_Y.append(2)
            tsi += 1

    return trn_X, trn_Y, tst_X, tst_Y

