import rbm
import pickle


class DBN(object):

    def __init__(self, input_size, dbn_shape):

        self._rbm_hidden_sizes = dbn_shape
        self._input_size = input_size
        self.rbm_list = []

        print("Construct DBN ...")
        for i, size in enumerate(dbn_shape):
            print('RBM {} : {} -> {}'.format(i, input_size, size))
            self.rbm_list.append(rbm.RBM(input_size, size))
            input_size = size

    def pre_train(self, x):
        for i, _rbm in enumerate(self.rbm_list):
            print("RBM {}: pre-training ...".format(i))
            _rbm.train(x)
            x = _rbm.rbm_output(x)

    def save_parameters(self):
        model = {
            'input_size': self._input_size,
            'rbm_hidden_sizes': self._rbm_hidden_sizes,
            'params': []
        }
        for _rbm in self.rbm_list:
            model['params'].append({'w': _rbm.w, 'hb': _rbm.hb})
        with open('./params/dbn_parameters.pkl', 'wb') as f:
            pickle.dump(model, f)
        print('pre training parameters saved !')


if __name__ == '__main__':
    dbn = DBN(784, [500, 200, 50])

