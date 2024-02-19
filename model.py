import tensorflow as tf
import numpy as np

from ops import log_clip

class LogisticRegression(object):
    def __init__(self, weight_decay):
        self.x, self.y, self.loss, self.hvp, self.train_grad, self.test_grad, self.w_ph, self.u = self._build(weight_decay)

    def _build(self, wd):
        x = tf.keras.Input(dtype=tf.float32, shape=(784,), name='input_image')
        y = tf.keras.Input(dtype=tf.float32, shape=(1,), name='grand_truth')

        w = tf.Variable(tf.zeros([784]), dtype=tf.float32, name='w')

        logits = tf.matmul(x, tf.reshape(w, [-1, 1]))
        preds = tf.nn.sigmoid(logits)
        train_loss = -tf.reduce_mean(y * log_clip(preds) + (1 - y) * log_clip(1 - preds)) + tf.nn.l2_loss(w) * wd
        test_loss = -tf.reduce_mean(y * log_clip(preds) + (1 - y) * log_clip(1 - preds))

        w_ph = tf.keras.Input(dtype=tf.float32, shape=w.get_shape(), name='w_placeholder')
        w.assign(w, w_ph)

        u = tf.keras.Input(dtype=tf.float32, shape=w.get_shape())

        with tf.GradientTape() as tape:
            # hessian vector product 
            first_grad = tape.gradient(train_loss, w)[0]
            elemwise_prod = first_grad * u
            hvp = tape.gradient(elemwise_prod, w)[0]

            # gradient
            train_grad = tape.gradient(train_loss, w)[0]
            test_grad = tape.gradient(test_loss, w)[0]

        return x, y, test_loss, hvp, train_grad, test_grad, w_ph, u

    def get_inverse_hvp_lissa(self, v, x, y, scale=10, num_samples=5, recursion_depth=1000, print_iter=100):

        inverse_hvp = None

        for i in range(num_samples):
            print('Sample iteration [{}/{}]'.format(i+1, num_samples))
            cur_estimate = v
            permuted_indice = np.random.permutation(range(len(x)))

            for j in range(recursion_depth):

                x_sample = x[permuted_indice[j]:permuted_indice[j]+1]
                y_sample = y[permuted_indice[j]:permuted_indice[j]+1]

                # get hessian vector product
                hvp = hvp.numpy(feed_dict={self.x: x_sample,
                                            self.y: y_sample,
                                            self.u: cur_estimate})

                # update hv
                cur_estimate = v + cur_estimate - hvp / scale

                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth {}: norm is {}".format(j, np.linalg.norm(cur_estimate)))

            if inverse_hvp is None:
                inverse_hvp = cur_estimate / scale
            else:
                inverse_hvp = inverse_hvp + cur_estimate / scale

        inverse_hvp = inverse_hvp / num_samples
        return inverse_hvp
