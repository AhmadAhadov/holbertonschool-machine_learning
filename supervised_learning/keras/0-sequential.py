import tensorflow as tf

def build_model(nx, layers, activations, lambtha, keep_prob):
    # 784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95
    model = tf.keras.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(tf.keras.layers.Dense(
                input_shape = (nx,),
                activation=activations[i],
                kernel_regularizer=tf.keras.regularizers.l2(l2=lambtha)
            ))
        else:
            model.add(tf.keras.layers.Dense(
                activation=activations[i],
                kernel_regularizer=tr.keras.regularizers.l2(l2=lambtha)
            ))
        if i < len(layers) - 1:
            model.add(tf.keras.layers.Dropout(1-keep_prob))
    return model
