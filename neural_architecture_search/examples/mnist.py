from tensorflow.examples.tutorials.mnist import input_data
from dbonas import Searcher, Trials


def create_model(hidden_size=128):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hidden_size, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model(x)

def objectve(trial):

    x = tf.placeholder(tf.float32, [None, 28, 28])
    model = create_model()
    y = model(x)
    a
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    sess = tf.InteractiveSession()

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    tf.initialize_all_variables().run()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ == '__main__':
    # tiral_config.register()
    searcher = Searcher()
    searcher.register_trial('batchsize', )
    searcher.register_trial('', )
    searcher.register_trial('lr', [0.1, 0.2])
    searcher.register_trial('optim', ['Adam', 'SGD'])
    searcher.search(objectve, n_trials=100)
    print('best_trial', searcher.best_trial)
    print('best_value', searcher.best_value)
