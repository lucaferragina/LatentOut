import numpy as np
from pyod.models import gmm, iforest, knn, lof, ocsvm
from tensorflow import keras
from VAE import Encoder, Decoder, VariationalAE
from tensorflow.keras.losses import binary_crossentropy
import cfof


def LatentOUT(x, architecture='VAE', ep=30, batch_size=64, layers_dim=None, score='knn', k=5):
    if architecture == 'VAE':
        if layers_dim is None:
            layers_dim = [x.shape[0], 2]

        encoder_inputs = keras.Input(shape=(layers_dim[0],))
        encoder = Encoder.build(layers_dim, encoder_inputs)
        layers_dim.reverse()
        latent_inputs = keras.Input(shape=(layers_dim[0],))
        decoder = Decoder.build(layers_dim, latent_inputs)
        model = VariationalAE(encoder, decoder)
        model.compile(optimizer=keras.optimizers.Adam())

        model.fit(x, epochs=ep, verbose=0, batch_size=batch_size)
        latent = np.array(model.encoder.predict(x)[0])
        reconstruction = np.array(model.decoder.predict(latent))
        base_score = np.array(binary_crossentropy(x, reconstruction))
        F_space = np.concatenate((latent, base_score.reshape(base_score.shape[0], 1)), axis=1)

    if score == 'cfof':
        clf = cfof.CFOF()
        LO_score = clf.fit_predict(F_space, k=k)

    if score == 'gmm':
        clf = gmm.GMM(n_components=k)
        clf.fit(x)
        LO_score = clf.decision_function(F_space)

    if score == 'if':
        clf = iforest.IForest()
        clf.fit(x)
        LO_score = clf.decision_function(F_space)

    if score == 'knn':
        clf = knn.KNN(n_neighbors=k)
        clf.fit(F_space)
        LO_score = clf.decision_function(F_space)

    if score == 'lof':
        clf = lof.LOF(n_neighbors=k)
        clf.fit(x)
        LO_score = clf.decision_function(F_space)

    if score == 'svm':
        clf = ocsvm.OCSVM(kernel=k)
        clf.fit(x)
        LO_score = clf.decision_function(F_space)

    return LO_score















