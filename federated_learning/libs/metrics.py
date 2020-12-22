import tensorflow as tf
from tensorflow import keras


class MultitaskBinaryCrossentropyWithMask(keras.losses.Loss):
    def call(self, y_true, model_out):
        logits = model_out[0]
        masks = model_out[1]
        losses = []
        for task in range(12):
            mask = tf.cast(masks[:, task], tf.bool)
            y_true_masked = tf.boolean_mask(y_true[:, task], mask)
            logits_masked = tf.boolean_mask(logits[:, task], mask)
            loss = tf.keras.losses.binary_crossentropy(
                y_true_masked, logits_masked, from_logits=False
            )
            losses.append(loss)
        loss = tf.stack(losses)
        return loss


class AUCMultitask(keras.metrics.AUC):
    def __init__(self, name="auc_multitask", task_number=0, **kwargs):
        super(AUCMultitask, self).__init__(name=name, **kwargs)
        self.task_number = task_number

    def update_state(self, y_true, y_pred, sample_weight=None):
        model_out = y_pred
        logits = model_out[0]
        masks = model_out[1]
        losses = []
        mask = tf.cast(masks[:, self.task_number], tf.bool)
        y_true_masked = tf.boolean_mask(y_true[:, self.task_number], mask)
        logits_masked = tf.boolean_mask(logits[:, self.task_number], mask)
        super(AUCMultitask, self).update_state(y_true_masked, logits_masked)

