from DanTFNN import DanTFnet
from DanTFNN import MnistData
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/mnist/", one_hot=True)
mnist_data = MnistData(mnist)

layer_list = [ 784,500,500,500,10]

tf_net = DanTFnet(layer_list, mnist_data)

tf_net.StartSession()

tf_net.Train(10,int(tf_net.data.data.train.num_examples/100),100)

print(tf_net.Test())


tf_net.SaveSess("mnist-500-500-500")

# tf_net.LoadSess("mnist-500-500-500")

run_x, run_y = tf_net.data.NextTrainBatch(1)

print("Prediction:") 
print(tf_net.sess.run(tf_net.Run(run_x))[0])
print("Actual:")
print(list(run_y[0]).index(1))