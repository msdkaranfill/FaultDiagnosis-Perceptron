##Import Data
#https://stackoverflow.com/questions/11955000/how-to-preserve-matlab-struct-when-accessing-in-python
import os
from os import listdir
from os.path import isfile, join
from scipy import io

def _check_keys(dict, label):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries, then delete
    unwanted data to access the data easily later on."""

    delkeys = ('__header__', '__version__', '__globals__')
    for key in dict.copy():
        if isinstance(dict[key], io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
        if key in delkeys:
            dict.pop(key)

    data_dict = {}
    #removing "bearing" key to get the data dictionaries only
    for bearing, data in dict.items():
        data_dict = data

    data_dict.update({"label": label})
    return data_dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem

    return dict


def load_single_example(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    if '\\' in filename:
        label = filename.split('\\')[1].split("_")[0]
        if label[0] == "\\":
            label.split("\\")
    else:
        label = filename.split("_")[0]

    return _check_keys(data, label)


#load datas
def load_datas(data_folder):
    """returns a list of dictionaries from data in .mat files in given folder directory,
    with added key 'label', taken value of its file name
    param: data_folder
    """
    files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    all_datas = [] #a list of dictionaries of datas
    for file_name in files:
        file_name = os.path.join(data_folder, file_name)
        data_dict = load_single_example(file_name)
        all_datas.append(data_dict)

    return all_datas