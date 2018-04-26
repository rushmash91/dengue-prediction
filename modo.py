from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dengue = pd.read_csv('nd.csv')
dengue = shuffle(dengue)
x, y = dengue.iloc[:, 1:6], dengue.iloc[:, -1]
x = x.as_matrix()
y = y.as_matrix()
y = y.T
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

total_len = X_train.shape[0]

# Parameters
learning_rate = 0.0001

training_epochs = 2000
batch_size = 10
display_step = 1


#dropout_rate = 0.1
# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 100 # 2nd layer number of features
n_hidden_3 = 100
n_hidden_4 = 44

n_input = X_train.shape[1]
n_classes = 1

# tf Graph input
x = tf.placeholder("float", [None, 5])
y = tf.placeholder("float", [None])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)    
    layer_1 = tf.nn.dropout(layer_1, 0.5)


    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, 0.5)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, 0.8)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    out_layer = tf.cast(out_layer, tf.float32)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer

cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        label_value = batch_y
        estimate = p
        err = label_value-estimate
        print("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=",
                "{:.9f}".format(avg_cost))
            print("[-------------------------------")
            for i in range(10):
                print ("label value:", label_value[i],
                    "estimated value:", estimate[i])
            print("===============================")

    print("Optimization Finished!")

    # Test model
    pred = tf.cast(pred, tf.float32)
    correct_prediction = tf.equal(tf.argmax(pred, axis=1, output_type=tf.int32), tf.cast(y, tf.int32))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test})*(10**4))

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)


# #restoring saved models
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "/tmp/model.ckpt")
#   print("Model restored.")
#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())
