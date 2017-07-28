import models_utils
import numpy as np

def test_multiclass_precision_should_return_():
    test_labels = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                   [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                   [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.float32)

    test_predictions = np.array([[0.5,0,0],[0.5,0,0],[0.5,0,0],[0,0.5,0],[0,0,0.5],
                        [0,0.5,0],[0,0.5,0],[0,0.5,0],[0,0.5,0],[0,0.5,0],
                        [0,0,0.5],[0,0,0.5],[0,0,0.5],[0.5,0,0],[0,0.5,0]],dtype=np.float32)

    prec = models_utils.precision_multiclass(test_predictions,test_labels, False)
    print(prec)


if __name__=='__main__':
    test_multiclass_precision_should_return_()