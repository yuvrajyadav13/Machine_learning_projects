from real_fake import *

saver.restore(sess, model_path)

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("0 stands for Fake and 1 stands for Real")

for i in range(93, 101):
        prediction_run = sess.run(prediction, feed_dict = {x: X[i].reshape(1,4)})
        accuracy_run = sess.run(accuracy, feed_dict = {x: X[i].reshape(1, 4), y_: Y[i].reshape(1,2)})
        print("Original Class: ",Y.iloc[i,1], " Predicted values:", prediction_run, "Accuracy : ", accuracy_run)
        

