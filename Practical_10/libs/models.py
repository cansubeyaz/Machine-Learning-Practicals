from sklearn.neural_network import MLPClassifier

def create_mpl( flags, nclasses ):
    hidden_layers = flags.hidden + [nclasses]
    print( hidden_layers )
    return MLPClassifier( hidden_layer_sizes=hidden_layers, activation='relu',
                          solver='adam', random_state=1, max_iter=flags.epochs, warm_start=True )