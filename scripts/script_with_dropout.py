# Load pickled data
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

dir = 'data/'

training_file = dir + 'train.p'
validation_file = dir + 'valid.p'
testing_file = dir + 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


from scipy.ndimage import shift
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import cv2

X_train, y_train = shuffle(X_train, y_train)

def rotate_image(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def generate_augmented_images(X_input, Y_input, augt_copies=10):
    shape = X_input[0].shape

    x_shift_range = int(shape[0] * 0.2) * 2
    y_shift_range = int(shape[1] * 0.2) * 2
    
    rotate_degrees = random.sample(range(-60, 60), augt_copies) 
    
    shift_values1 = (np.random.sample(augt_copies) - 0.5) * x_shift_range
    shift_values2 = (np.random.sample(augt_copies) - 0.5) * y_shift_range

    dict1 = zip(shift_values1, shift_values2)
    
    X_augmented = []
    Y_augmented = []
   
    for idx, x in enumerate(X_input):    
        
        for degree in rotate_degrees:
            rot1 = rotate_image(x, degree) 
            X_augmented.append(rot1)
            Y_augmented.append(Y_input[idx])
    
        for v1, v2 in dict1:
            s1 = shift(x,[v1,v2,0],mode='nearest')
            X_augmented.append(s1)
            Y_augmented.append(Y_input[idx])      
    
    return np.append(X_input,X_augmented,axis=0), np.append(Y_input,Y_augmented,axis=0)

def normalize(input):
    arr = [cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in input]
    return np.array(arr)

def grayscale(input):
    arr = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  for img in input]
    return np.array(arr)

def pre_process(X):
    X_train_use = grayscale(X)
    X_train_use = normalize(X_train_use)
    X_train_use = np.reshape(X_train_use, (-1, 32, 32, 1))

    return X_train_use

### Define model architecture

#define hyper parameters
EPOCHS = 15
BATCH_SIZE = 64

num_channel = 1
num_classes = 43
learning_rate = 0.001

# define ConvNet Architecture
def ConvNet(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x128.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, num_channel, 128), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(128))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 32x32x128. Output = 16x16x128.
    conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    

    # Layer 2: Convolutional. Input = 16x16x128. Output = 16x16x128.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 128, 128), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(128))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 16x16x128. Output = 8x8x128.
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    
    # Layer 3: Convolutional. Input = 8x8x128. Output = 8x8x128.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 128, 128), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(128))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    conv3 = tf.nn.relu(conv3)
    # Pooling. Input = 8x8x128. Output = 4x4x128.
    conv3 = tf.nn.avg_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Layer 4: Convolutional. Input = 4x4x128. Output = 4x4x128.
    conv4_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 128, 128), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(128))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
    conv4 = tf.nn.relu(conv4)
    # Pooling. Input = 4x4x128. Output = 2x2x128.
    conv4 = tf.nn.avg_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.dropout(conv4, keep_prob)

    # Flatten. Input = 2x2x128. Output = 512.
    fc0   = flatten(conv4)
    
    # Layer 5: Fully Connected. Input = 512. Output = 1024.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(512, 1024), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)

    # Layer 6: Fully Connected. Input = 1024. Output = 1024.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(1024))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)

    # Layer 7: Fully Connected. Input = 1024. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1024, num_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(num_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
tf.reset_default_graph()

#define placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, num_channel), name='x')
y = tf.placeholder(tf.int32, (None), name='y')
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

logits = ConvNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        
        batch_x = pre_process(batch_x)
        
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
   
    print("Training...")
    print()
    for i in range(EPOCHS):
        
        X_train, y_train = shuffle(X_train, y_train)

        batch_counter = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            
            batch_x, batch_y = generate_augmented_images(X_train[offset:end], y_train[offset:end],augt_copies=20)
            batch_x = pre_process(batch_x)
            
            batch_x, batch_y = shuffle(batch_x,batch_y)
            
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.85})
            batch_counter += 1
            if batch_counter % 50 == 0:
                print ('batch number: '+str(batch_counter))
                batch_validation_accuracy = evaluate(X_valid, y_valid)
                print("Batch Validation Accuracy = {:.3f}".format(batch_validation_accuracy))
                print()
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
       
    saver.save(sess, './model_new/traffic_sign')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model_new'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))