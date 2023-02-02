import pickle

def save_model(filename, object):
    with open(f"{filename}", 'wb') as file:
        pickle.dump(object, file)
        
def load_fraud_model(filename):
    with open(f'{filename}', 'rb') as file:
        model = pickle.load(file)
    return model