import tensorflow as tf


def make_parallel(model, num_copies, num_gpus):
    # num of copies of model
    model.n = num_copies

    # keep track of placholder and gradients
    tower_grads, ph, losses, acc = [], [], [], []

    with tf.device('/cpu:0'):
        with tf.variable_scope(tf.get_variable_scope()):
            # number of copies
            for i in range(num_gpus):
                with tf.device('/gpu:{}'.format(i)):
                    for j in range(num_copies//num_gpus):
                        with tf.name_scope('gpu_{}_{}'.format(i,j)) as scope:
                            # run inference
                            #  get handles to placeholders, logits and probs
                            placeholders, logits = model.inference()

                            # get loss, accuracy
                            loss, accuracy = model.compute_loss(placeholders, logits)

                            # reuse trainable parameters
                            tf.get_variable_scope().reuse_variables()

                            # gather gradients
                            grads = model.opt.compute_gradients(loss)

                            # save grads for averaging later
                            tower_grads.append(grads)

                            # save the list of placholder handles
                            ph.append(list(placeholders.values()))

                            # save loss and accuracy
                            losses.append(loss)
                            acc.append(accuracy)

        # average gradients
        grads = average_gradients(tower_grads)
        # apply averaged gradients
        apply_gradient_op = model.opt.apply_gradients(grads)

        # attach to instance
        model.train_op = apply_gradient_op
        # losses
        model.loss = tf.reduce_mean(losses)
        model.accuracy = tf.reduce_mean(acc)

        # attach placeholders to instance
        model.placeholders = ph

    return 


# from cifar10 multi-gpu example
#  https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
