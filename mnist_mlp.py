import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

# hyper-parameters
DATA_FNAME = 'mnist_traindata.hdf5'
MODEL_FNAME = 'hw3p2.hdf5'
epoch = 50
batch_size = 100


class activation_func:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        x = x - max(x)
        exp_x = np.exp(x)
        return exp_x/np.sum(exp_x)


class Loss:
    @staticmethod
    def cross_entropy(x, y):
        """"
        :param x: softmax output at last layer
        :param y: ground truth vector (assume to be one-hot)
        :return:
        """
        return np.sum(-np.log(x+0.000001)*y)


class SGD:
    def __init__(self, model, activation='relu', lr=0.1):
        self.model = model
        self.activation = activation
        self.lr = lr

    def relu_grad(self, x):
        if x < 0:
            grad = 0
        elif x == 0:
            grad = 0.5
        else:
            grad = 1
        return grad

    def relu_grad_vec(self, x):
        grad = []
        for i in range(len(x)):
            grad.append(self.relu_grad(x[i]))
        return np.diag(grad)

    def tanh_grad(self, x):
        return 1-np.tanh(x)**2

    def tanh_grad_vec(self, x):
        grad = []
        for i in range(len(x)):
            grad.append(self.tanh_grad(x[i]))
        return np.diag(grad)

    def softmax_crossentropy_grad(self, x, y):
        """
        :param x: output at last layer before softmax
        :param y: ground truth vector (assume to be one-hot)
        :return:
        """
        x -= max(x)
        exp_x = np.exp(x)
        sum_exp = np.sum(exp_x)
        grad = exp_x/sum_exp-y
        return grad

    ### CORE PART
    def compute_gradient(self, data_x, data_y):
        # forward computing
        _, loss = self.model.forward(data_x, data_y)
        # back propagation
        self.delta = []
        self.delta.append(self.softmax_crossentropy_grad(self.model.layers_s[-1], data_y))
        for i in range(2, self.model.n_layers):
            dc_da = np.dot(self.delta[0], self.model.layers_W[-i+1].T)
            if self.activation == 'relu':
                da_ds = self.relu_grad_vec(self.model.layers_s[-i])
            elif self.activation == 'tanh':
                da_ds = self.tanh_grad_vec(self.model.layers_s[-i])
            else:
                exit('unknown activation function')
            dc_ds = np.dot(dc_da, da_ds)
            self.delta.insert(0, dc_ds)
        for i in range(self.model.n_layers-1):
            (dim1, dim2) = self.model.layers_W[i].shape
            ds_dw = np.zeros((dim2, dim1, dim2))
            for d in range(dim2):
                ds_dw[d, :, d] = self.model.layers_a[i - 1] if i > 0 else data_x
            grad_W = np.sum(self.delta[i]*ds_dw, axis=0)
            if len(self.gradient_W) < self.model.n_layers-1:
                self.gradient_W.append(grad_W)
                self.gradient_b.append(self.delta[i])
            else:
                self.gradient_W[i] += grad_W
                self.gradient_b[i] += self.delta[i]
        return loss

    def compute_gradient_batch(self, data_x_batch, data_y_batch):
        self.gradient_W = []
        self.gradient_b = []
        loss_buff = []
        for i in range(batch_size):
            loss = self.compute_gradient(data_x_batch[i], data_y_batch[i])
            loss_buff.append(loss)
        self.gradient_W = np.array(self.gradient_W)/batch_size
        self.gradient_b = np.array(self.gradient_b)/batch_size
        return loss_buff

    def update(self, data_x_batch, data_y_batch):
        """
        :param data_x: input vector
        :param data_y: ground truth vector
        :return:
        """
        loss_buff = self.compute_gradient_batch(data_x_batch, data_y_batch)
        self.model.layers_W -= self.gradient_W * float(self.lr)
        self.model.layers_b -= self.gradient_b * float(self.lr)
        return loss_buff

class MLP:
    def __init__(self, n_layers, size, activation='relu'):
        """
        :param n_layers: number of layers
        :param size:
        :param input_size:
        """
        assert len(size) == n_layers
        self.n_layers = n_layers
        self.size = size
        self.activation = activation

    def setup(self):
        """
        initialization all parameters
        :return:
        """
        self.layers_W = []
        self.layers_b = []
        self.layers_s = []
        self.layers_a = []
        for i in range(self.n_layers-1):
            self.layers_W.append(
                np.random.normal(size=(self.size[i], self.size[i+1])))
            self.layers_b.append(
                np.random.normal(size=(self.size[i+1], )))
            self.layers_s.append(
                np.zeros(self.size[i+1]))
            self.layers_a.append(
                np.zeros(self.size[i+1]))

    def forward(self, input, ground_truth):
        # first output
        s = np.dot(input, self.layers_W[0]) + self.layers_b[0]
        if self.activation == 'relu':
            a = activation_func.relu(s)
        elif self.activation == 'tanh':
            a = activation_func.tanh(s)
        else:
            exit('unknown activation function')
        self.layers_s[0] = s
        self.layers_a[0] = a

        for i in range(1, self.n_layers-2):
            s = np.dot(a, self.layers_W[i])+self.layers_b[i]
            if self.activation == 'relu':
                a = activation_func.relu(s)
            elif self.activation == 'tanh':
                a = activation_func.tanh(s)
            else:
                exit('unknown activation function')
            self.layers_s[i] = s
            self.layers_a[i] = a

        # last output
        s = np.dot(a, self.layers_W[-1]) + self.layers_b[-1]
        a = activation_func.softmax(s)
        self.layers_s[-1] = s
        self.layers_a[-1] = a
        prediction = np.where(a==max(a))[0][0]
        loss = Loss.cross_entropy(a, ground_truth)
        return prediction, loss

    def save_model(self, model_fname):
        with h5py.File(model_fname, 'w') as hf:
            hf.attrs['act'] = np.string_(self.activation)
            for i in range(len(self.layers_W)):
                hf.create_dataset('w{}'.format(i + 1), data=self.layers_W[i])
                hf.create_dataset('b{}'.format(i + 1), data=self.layers_b[i])

def main():
    # accept parameters for command lines
    assert len(sys.argv) == 3
    lr =float(sys.argv[1])
    activation = sys.argv[2]

    with h5py.File(DATA_FNAME, 'r') as hf:
        xdata = hf['xdata'][:]
        ydata = hf['ydata'][:]
    x_train = xdata[:10000]
    y_train = ydata[:10000]
    x_valid = xdata[-1000:]
    y_valid = ydata[-1000:]
    assert x_train.shape[1]==784
    assert y_train.shape[1]==10

    iterates = int(x_train.shape[0]/batch_size)
    mlp = MLP(n_layers=3, size=[784, 100, 10], activation=activation)
    mlp.setup()
    optimizer = SGD(model=mlp, activation=activation, lr=lr)
    train_loss = []
    train_accu = []
    valid_accu = []
    max_valid_accu = 0

    for epo in range(epoch):
        if (epo+1)%20==0:
            lr = lr / 2.0
        print('----------epoch {}----------'.format(epo))
        for iter in range(iterates):
            x_batch = x_train[iter*batch_size:(iter+1)*batch_size]
            y_batch = y_train[iter*batch_size:(iter+1)*batch_size]
            loss_buff = optimizer.update(x_batch, y_batch)
            loss = np.mean(loss_buff)
            train_loss.append(loss)
            print('epoch{}\titerates={}\tloss={}'.format(epo, iter, loss))

        # accuracy on training/validation data
        correct_train = 0
        for i in range(x_train.shape[0]):
            pred, _ = mlp.forward(x_train[i], y_train[i])
            if pred == np.where(y_train[i] == max(y_train[i]))[0][0]:
                correct_train += 1
        train_accu.append(correct_train / x_train.shape[0])

        correct_valid = 0
        for i in range(x_valid.shape[0]):
            pred, _ = mlp.forward(x_valid[i], y_valid[i])
            if pred == np.where(y_valid[i]==max(y_valid[i]))[0][0]:
                correct_valid += 1
        valid_accu.append(correct_valid / x_valid.shape[0])
        print('train correct = {}\tvalid correct = {}'.format(correct_train, correct_valid))
        if epo == 0:
            max_valid_accu = valid_accu[-1]
            mlp.save_model(MODEL_FNAME)
        else:
            if valid_accu[-1] > max_valid_accu:
                max_valid_accu = valid_accu[-1]
                mlp.save_model(MODEL_FNAME)

    print(train_accu)
    print(valid_accu)
    plt.figure()
    plt.title('initial learning rate = {}, activation: {}'.format(lr, activation))
    plt.subplot(211)
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.plot(train_loss)
    plt.subplot(212)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.plot(train_accu, label='training')
    plt.plot(valid_accu, label='validation')
    plt.legend(loc='upper right')
    plt.savefig('{}_{}.png'.format(activation, lr))
    plt.show()


if __name__ == '__main__':
    main()