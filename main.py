import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import config
import resnet_blocks
from split_dataset import SplitDataset
from prepare_data import generate_datasets

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


split_dataset = SplitDataset(dataset_dir="original_dataset",
                             saved_dataset_dir="dataset",
                             show_progress=True)
split_dataset.start_splitting()

train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()


model = resnet_blocks.ResNet152([3,8,36,3])
model.build(input_shape=(224,224,3))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


train_acc_metric  = tf.keras.metrics.SparseCategoricalAccuracy()

val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
print('\nFinished preparing data.')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss_fn(labels, logits)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(labels, logits)
    return loss_value


@tf.function
def test_step(images, labels):
    val_logits = model(images, training=False)
    val_acc_metric.update_state(labels, val_logits)
    test_loss = loss_fn(labels, val_logits)
    return test_loss


# start training
for epoch in range(config.EPOCHS):

    step = 0
    for images, labels in train_dataset:
        step += 1
        train_loss = train_step(images, labels)

        print(f"\rEpoch: {epoch + 1}/{config.EPOCHS}, Step: {step}/{math.ceil(train_count / config.BATCH_SIZE)}, Train loss: {train_loss}, Train accuracy: {train_acc_metric.result():.5f}", end="")

    for valid_images, valid_labels in valid_dataset:
        test_loss = test_step(valid_images, valid_labels)
    print('\n')
    print(f"Epoch: {epoch + 1}/{config.EPOCHS}, train loss: {train_loss}, train accuracy: {train_acc_metric.result()},"
          f"valid loss: {test_loss}, valid accuracy: {val_acc_metric.result()}")
    print('\n')
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()







