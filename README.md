## RNA State Inference with LSTMs

Devin Willmott, Dr. David Murrugarra, Dr. Qiang Ye

---

License
-----

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Required Software and Packages
------
* python 3.6
* numpy 1.14.3
* keras 2.1.5
* theano 0.9.0 or tensorflow 1.4.1
* gtfold 1.16
* (Optional) CUDA 8.0 and cuDNN 5.1

Instructions
------

1) Clone this repository, install all required packages above.

2) Get data, available at [http://ms.uky.edu/~dwi239/rnastateinf-data.zip]. This will contain training and testing data for the neural network and HMM in the 'data' directory, and the output of the neural network used to direct NNTM in the 'probabilities' directory. Unzip this into the cloned directory.

State Inference
------

Type

```python rnn.py```

to begin training and testing a neural network with the architecture and hyperparameters from the paper, or use the command line arguments to choose your own. For example, to train and test an RNN with a learning rate of 0.01 and a network with three bidirectional LSTM hidden layers of sizes 300, 200, and 100, run

```python rnn.py --lr 0.01 --hiddensizes 300 200 100```

The HMM has two command line arguments: the mode, and k, the order of the HMM. There are three modes: 'train' trains and saves an HMM of order k, 'run' uses a saved HMM of order k to perform inference on the test set, and 'cycle' does train' and 'run' using all orders of HMM from 1 to k. So, to train an order 4 HMM, type

```python hmm.py 4 train```

Predicted State Directed NNTM
------

The method.py file takes the output of the neural network and uses it to perform state directed NNTM. With the neural network output in the directory names 'probabilities', running

```python method.py```

will generate SHAPE and produce three predicted structures for each sequence: no SHAPE direction, native state direction, and predicted state direction. It will print PPV, sensitivity, and accuracy of all three predictions.