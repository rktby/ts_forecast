import tensorflow as tf
import numpy as np

def train_model(model, data_iterator, optimizer, hparams, iterations=10, print_each=1):
    for epoch in range(10):
        losses = []
        model.training = not model.training
        for inp, inp_max, inp_min, target, target_max, target_min, cond in iter(data_iterator):
            with tf.GradientTape() as tape:
                pred = model(inp, None)

                loss = calculate_loss(pred, target, model, hparams)
                total_loss = tf.reduce_sum(loss)

            # Update gradients
            variables = model.variables
            gradients = tape.gradient(total_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            losses.append(total_loss)

        optimizer._lr *= hparams.lr_decay
        if epoch % print_each == 0:
            print(epoch, ': ', np.round(np.dstack(losses).mean(), 5), np.round(loss, 5))

def calculate_loss(pred, target, model, hparams):
    loss = []
    for n, (p, (field_name, is_embedded)) in enumerate(zip(pred, model.target['processing'])):
        t = target[:,:,n:n+1]
        if is_embedded:
            t = t * model.channels[field_name]
            t = tf.cast(t, tf.int32)
            err = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p, labels=t[:,:,0]) / 100
        else:
            err = tf.losses.mean_squared_error(t, p)
        err = tf.reduce_mean(err)
        loss.append(err)

    l2_loss = [tf.reduce_sum(var ** 2) * ('bias' not in var.name) for var in model.trainable_variables]
    l2_loss = tf.reduce_sum(l2_loss) * hparams.lambd
    loss.append(l2_loss)

    return loss