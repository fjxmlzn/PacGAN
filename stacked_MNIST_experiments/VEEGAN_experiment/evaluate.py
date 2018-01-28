import theano

from scipy import misc
import numpy

import keras
model = keras.models.load_model("mnist_model.hdf5")

def evaluate(x):
    output = model.predict(x)
    return list(numpy.argmax(output, axis=1))
    
if __name__ == "__main__":
    p2 = numpy.reshape(misc.imread("2.png"), (1, 28, 28, 1))
    p9 = numpy.reshape(misc.imread("9.png"), (1, 28, 28, 1))
    
    print(evaluate(numpy.concatenate((p2, p9), axis=0)))