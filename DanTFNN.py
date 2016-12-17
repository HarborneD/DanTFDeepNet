import tensorflow as tf

class TFReadyData():

	def __init__(self,data):
		#should store some reference to the data (TF reference, file path etc. This will depend on the intended access methods)
		self.data=data
		

	def NextTrainBatch(self,batch_size):
		#should return x and y data for next batch of training data
		pass

	def NextTestBatch(self):
		#should return x and y data for next batch of test data
		pass


class MnistData(TFReadyData):
	def __init__(self,data):
		self.data=data
		

	def NextTrainBatch(self,batch_size):
		return self.data.train.next_batch(batch_size)


	def NextTestBatch(self):

		return self.data.test.images, self.data.test.labels


class DanTFnet():

	def __init__(self, layersize_list, tf_ready_data):
		self.X,self.Y_,self.W_list,self.B_list,self.y_list = self.CreateTFVariables(layersize_list)
		self.data = tf_ready_data

		self.saver = tf.train.Saver()

	def CreateTFVariables(self,layer_list):
		X = tf.placeholder(tf.float32, [None,layer_list[0]])
		Y_ = tf.placeholder(tf.float32, [None, layer_list[-1]]) 

		W_list = []
		B_list = []

		for layer_size_index in range(1,len(layer_list)):
			W_list.append( tf.Variable(tf.random_normal([layer_list[layer_size_index-1], layer_list[layer_size_index]])) )
			B_list.append( tf.Variable(tf.random_normal([layer_list[layer_size_index]])) )

		y_list = []

		y_list.append(X)

		for layer_index in range(len(W_list)-1):
			y_list.append(  tf.add( tf.matmul(y_list[-1], W_list[layer_index]), B_list[layer_index] ) ) 
			y_list[-1] = tf.nn.relu(y_list[-1])

		y_list.append(  tf.add( tf.matmul(y_list[-1], W_list[-1]), B_list[-1] ) ) 

		return X,Y_,W_list,B_list,y_list


	def StartSession(self):
		self.sess = tf.Session()
		

	def CloseSession(self):
		self.sess.close()


	def Train(self,epochs,n_batches,batch_size):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_list[-1], self.Y_))
		#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
		train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

		tf.global_variables_initializer().run(session=self.sess)	
		
		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(n_batches):
				batch_xs, batch_ys = self.data.NextTrainBatch(batch_size)
				_,bloss =  self.sess.run([train_step,cross_entropy], feed_dict={self.X: batch_xs, self.Y_: batch_ys})
				epoch_loss += bloss
			print(epoch_loss)


	def Test(self):
		
		correct_prediction = tf.equal(tf.argmax(self.y_list[-1], 1), tf.argmax(self.Y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		test_x,test_y = self.data.NextTestBatch()
		return self.sess.run( accuracy, feed_dict={self.X: test_x, self.Y_: test_y})


	def Run(self,x_input):
		y_list = []

		y_list.append(x_input)

		for layer_index in range(len(self.W_list)-1):
			y_list.append(  tf.add( tf.matmul(y_list[-1], self.W_list[layer_index]), self.B_list[layer_index] ) ) 
			y_list[-1] = tf.nn.relu(y_list[-1])

		return tf.argmax(tf.add( tf.matmul(y_list[-1], self.W_list[-1]), self.B_list[-1] ),1)


	def SaveSess(self,save_path):
		self.saver.save(self.sess, save_path)


	def LoadSess(self,load_path):
		self.saver.restore(self.sess, load_path)



