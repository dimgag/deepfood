# 
import time 
import tensorflow as tf
# tf.config.run_functions_eagerly(False)

# SAM Model
# class SAMModel(tf.keras.Model):
#     def __init__(self, resnet_model, rho=0.05):
#         """
#         p, q = 2 for optimal results as suggested in the paper
#         (Section 2)
#         """
#         super(SAMModel, self).__init__()
#         self.resnet_model = resnet_model
#         self.rho = rho

#     def train_step(self, data):
#         (images, labels) = data
#         e_ws = []
#         with tf.GradientTape() as tape:
#             predictions = self.resnet_model(images)
#             loss = self.compiled_loss(labels, predictions)
#         trainable_params = self.resnet_model.trainable_variables
#         gradients = tape.gradient(loss, trainable_params)
#         grad_norm = self._grad_norm(gradients)
#         scale = self.rho / (grad_norm + 1e-12)

#         for (grad, param) in zip(gradients, trainable_params):
#             e_w = grad * scale
#             param.assign_add(e_w)
#             e_ws.append(e_w)

#         with tf.GradientTape() as tape:
#             predictions = self.resnet_model(images)
#             loss = self.compiled_loss(labels, predictions)    
        
#         sam_gradients = tape.gradient(loss, trainable_params)
#         for (param, e_w) in zip(trainable_params, e_ws):
#             param.assign_sub(e_w)
        
#         self.optimizer.apply_gradients(
#             zip(sam_gradients, trainable_params))
        
#         self.compiled_metrics.update_state(labels, predictions)
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, data):
#         (images, labels) = data
#         predictions = self.resnet_model(images, training=False)
#         loss = self.compiled_loss(labels, predictions)
#         self.compiled_metrics.update_state(labels, predictions)
#         return {m.name: m.result() for m in self.metrics}

#     def _grad_norm(self, gradients):
#         norm = tf.norm(
#             tf.stack([
#                 tf.norm(grad) for grad in gradients if grad is not None
#             ])
#         )
#         return norm






# <-------------------------------------------------------------------------------------------------------------->
class SAMModel(tf.keras.Model):
    def __init__(self, model, rho=0.05):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.model = model
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.compiled_loss(labels, predictions)    
        
        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
        
        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))
        
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (images, labels) = data
        predictions = self.model(images, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm
