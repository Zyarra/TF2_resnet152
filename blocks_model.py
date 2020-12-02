import tensorflow as tf
from config import NUM_CLASSES


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3),
                                            strides=1,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1), strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1),
                                            strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1),
                                            strides=stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1),
                                            strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layers(filter_num, num_blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, num_blocks):
        res_block.add(BasicBlock(filter_num, stride=stride))

    return res_block


def make_bottleneck_layers(filter_num, num_blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(Bottleneck(filter_num, stride=stride))

    for _ in range(1, num_blocks):
        res_block.add(Bottleneck(filter_num, stride=1))

    return res_block


class ResNet152(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNet152, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7),
                                            strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2,
                                               padding='same')

        self.layer1 = make_basic_block_layers(filter_num=64, num_blocks=layer_params[0])
        self.layer2 = make_basic_block_layers(filter_num=128, num_blocks=layer_params[1],
                                              stride=2)
        self.layer3 = make_basic_block_layers(filter_num=256, num_blocks=layer_params[2],
                                              stride=2)
        self.layer4 = make_basic_block_layers(filter_num=512, num_blocks=layer_params[3],
                                              stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
        self.flat = tf.keras.layers.Flatten()
        self.d = tf.keras.layers.Dense(units=1024, activation='relu')
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
        self.drop = tf.keras.layers.Dropout(0.1)



    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.bn2(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        x = self.bn3(x)
        x = self.flat(x)
        x = self.d(x)
        x = self.drop(x)
        output = self.fc(x)

        return output
