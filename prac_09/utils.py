import pickle

def save_model(model, filename, mean, std, threshold):
    dit = {
        'model': model,
        'mu': mean,
        'std': std,
        'thresh': threshold
    }
    with open(filename, 'wb') as fid:
        pickle.dump(dit, fid)

def load_model(filename):
    with open(filename, 'rb') as fid:
        return pickle.load(fid)
