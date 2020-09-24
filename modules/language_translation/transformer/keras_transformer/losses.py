import keras

def masked_ce(layer_size, homoscedastic=False):

    def _variance_minimization(loss):
        loss = keras.backend.cast(loss, 'float32')
        variance = keras.backend.var(loss)
        mean = keras.backend.mean(loss)
        precision = keras.backend.exp(-variance)
        return precision, mean

    def _homoscedastic_masked_ce(y_true, y_pred):
        raise NotImplementedError("Homoscedastic uncertainty wrapper not implemented yet")

    def _masked_ce(y_true, y_pred):
        y_true = keras.backend.tf.reshape(y_true[:, 1:], [-1,layer_size-1])
        y_true = keras.backend.cast(y_true, 'int32')
        loss = keras.backend.tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        mask = keras.backend.tf.cast(keras.backend.tf.not_equal(y_true, 0), 'float32')
        loss = keras.backend.tf.reduce_sum(loss * mask, -1) / keras.backend.tf.reduce_sum(mask, -1)
        loss = keras.backend.mean(loss)
        return loss

    if homoscedastic:
        return _homoscedastic_masked_ce
    else:
        return _masked_ce