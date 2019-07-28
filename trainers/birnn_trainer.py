import tensorflow as tf
import numpy as np

def train_model(model, train_data, val_data, optimizer, hparams, iterations=10, print_each=1, alternate_dropout=True):
    for epoch in range(iterations):
        losses = []
        if alternate_dropout:
            model.training = not model.training
        # Iterate over training datasets and update model
        for inp, inp_max, inp_min, target, target_max, target_min, cond in iter(train_data):
            train_loss = train_step(model, inp, target, optimizer, hparams)
            losses.append(train_loss)

        # Calculate validation loss
        inp, inp_max, inp_min, target, target_max, target_min, cond = next(iter(val_data))
        val_loss = calculate_loss(inp, target, model, hparams)
        
        # Update optimizer
        optimizer._lr *= hparams.lr_decay
        
        # Print loss
        if epoch % print_each == 0:
            print(epoch, ': ', np.round(np.dstack(losses).mean(), 5), np.round(train_loss, 5), np.round(val_loss, 5))

def train_step(model, inp, target, optimizer, hparams):
    # Calculate loss
    with tf.GradientTape() as tape:
        train_loss = calculate_loss(inp, target, model, hparams)
        train_loss = tf.reduce_sum(train_loss)

    # Update gradients
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_loss
                
def calculate_loss(inp, target, model, hparams):
    pred = model(inp, None)

    loss = []
    for n, (p, (field_name, is_embedded)) in enumerate(zip(pred, model.target['processing'])):
        # Iterate through each output field
        t = target[:,:,n:n+1]
        # Calculate cross entropy loss if output is categorical embedded data
        if is_embedded:
            t = t * model.channels[field_name]
            t = tf.cast(t, tf.int32)
            err = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p, labels=t[:,:,0]) / 100
        # Calculate MSE loss if output is continuous data
        else:
            err = tf.losses.mean_squared_error(t, p)
        err = tf.reduce_mean(err)
        loss.append(err)

    # Calculate L2 loss
    l2_loss = [tf.reduce_sum(var ** 2) * ('bias' not in var.name) for var in model.trainable_variables]
    l2_loss = tf.reduce_sum(l2_loss) * hparams.lambd
    loss.append(l2_loss)

    return loss
