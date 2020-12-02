import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import config


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False)

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model = tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.15))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(200, activation='softmax'))


train_d = tf.keras.preprocessing.image_dataset_from_directory('./dataset/train', batch_size=config.BATCH_SIZE, image_size=(config.image_height, config.image_width), shuffle=True, color_mode='rgb', labels='inferred')
test_d = tf.keras.preprocessing.image_dataset_from_directory('./dataset/valid', batch_size=config.BATCH_SIZE, image_size=(config.image_height, config.image_width), color_mode='rgb', labels='inferred)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=100,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

model.fit(train_d, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=1, validation_data=test_d, workers=16, callbacks=[early_stop]) #validation_steps=4,

tf.keras.models.save_model('efficient_bird.h5')
model.save_weights('bird_weights.hdf5')

#
#
#
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
#
# train_acc_metric  = tf.keras.metrics.SparseCategoricalAccuracy()
#
# val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
# print('\nFinished preparing data.')
#
#
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         logits = model(images, training=True)
#         loss_value = loss_fn(labels, logits)
#         loss_value += sum(model.losses)
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     train_acc_metric.update_state(labels, logits)
#     return loss_value
#
#
# @tf.function
# def test_step(images, labels):
#     val_logits = model(images, training=False)
#     val_acc_metric.update_state(labels, val_logits)
#     test_loss = loss_fn(labels, val_logits)
#     return test_loss
#
#
# # start training
# for epoch in range(config.EPOCHS):
#
#     step = 0
#     for images, labels in train_dataset:
#         step += 1
#         train_loss = train_step(images, labels)
#
#         print(f"\rEpoch: {epoch + 1}/{config.EPOCHS}, Step: {step}/{math.ceil(train_count / config.BATCH_SIZE)}, Train loss: {train_loss}, Train accuracy: {train_acc_metric.result():.5f}", end="")
#
#     for valid_images, valid_labels in valid_dataset:
#         test_loss = test_step(valid_images, valid_labels)
#     print('\n')
#     print(f"Epoch: {epoch + 1}/{config.EPOCHS}, train loss: {train_loss}, train accuracy: {train_acc_metric.result()},"
#           f"valid loss: {test_loss}, valid accuracy: {val_acc_metric.result()}")
#     print('\n')
#     train_acc_metric.reset_states()
#     val_acc_metric.reset_states()
#
# model.save_weights(filepath=config.save_model_dir, save_format='tf')
#
#
#
#
#
#
# #
# # class TokenAndPositionEmbedding(layers.Layer):
# #     def __init__(self, maxlen, vocab_size, embed_dim):
# #         super(TokenAndPositionEmbedding, self).__init__()
# #         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
# #         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
# #
# #     def call(self, x):
# #         maxlen = tf.shape(x)[-1]
# #         positions = tf.range(start=0, limit=maxlen, delta=1)
# #         positions = self.pos_emb(positions)
# #         x = self.token_emb(x)
# #         return x + positions
# #
#
#
#
# # class VisionTransformer(tf.keras.Model):
# #     def __init__(self, image_size, patch_size, num_layers, num_classes, d_model,
# #                  num_heads, dnn_dim, channels=3, dropout=0.1):
# #         super(VisionTransformer, self).__init__()
# #         num_patches = image_size // patch_size ** 2
# #
# #         self.patch_dim = channels * patch_size ** 2
# #         self.d_model = d_model
# #         self.num_layers = num_layers
# #         self.patch_size = patch_size
# #
# #         self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255.0)
# #         self.pos_emb = self.add_weight(name='class_embed', shape=(1, num_patches + 1, d_model))
# #         self.patch_proj = tf.keras.layers.Dense(d_model)
# #         self.enc_layers = [TransformerBlock(d_model, num_heads, dnn_dim, dropout) for _ in range(num_layers)]
# #         self.DNN_head = tf.keras.Sequential([
# #             tf.keras.layers.LayerNormalization(epsilon=1e-6),
# #             tf.keras.layers.Dense(dnn_dim, activation=tf.keras.activations.gelu),
# #             tf.keras.layers.Dropout(dropout),
# #             tf.keras.layers.Dense(num_classes),
# #         ])
# #
# #     def extract_patches(self, images):
# #         batch_size = tf.shape(images)[0]
# #         patches = tf.image.extract_patches(images=images, sizes=[1, self.patch_size, self.patch_size, 1],
# #                                            strides=[1, self.patch_size, self.patch_size, 1],
# #                                            rates=[1, 1, 1, 1],
# #                                            padding='VALID')
# #         patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
# #         return patches
# #
# #     def call(self, x, training, **kwargs):
# #         batch_size = tf.shape(x)[0]
# #         x = self.rescale(x)
# #         patches = self.extract_patches(x)
# #         x = self.patch_proj(patches)
# #
# #         class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
# #         x = tf.concat([class_emb, x], axis=1)
# #         x = x + self.pos_emb
# #
# #         for layer in self.enc_layers:
# #             x = layer(x, training)
# #
# #         x = self.DNN_head(x[:, 0])
# #         return x
# #
# #
# #
# #
# # model = VisionTransformer(image_size=image_size, patch_size=patch_size, num_layers=num_layers,
# #                           num_classes=1000, d_model=d_model, num_heads=num_heads, dnn_dim=dnn_dim,
# #                           channels=3, dropout=0.1)
# #
# #
# #
# # from split_dataset import SplitDataset
# #
# # split_dataset = SplitDataset(dataset_dir="original_dataset",
# #                              saved_dataset_dir="dataset",
# #                              show_progress=True)
# # split_dataset.start_splitting()