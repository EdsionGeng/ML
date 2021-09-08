import numpy as np
import tensorflow.compat.v1 as tf

docs = ['Well Done',
        'Good Work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
vocab_size = 50
encoded_docs = [tf.keras.preprocessing.text.one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# define the model
input = tf.keras.layers.Input(shape=(4,))
x = tf.keras.layers.Embedding(vocab_size, 8, input_length=max_length)(input)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=input, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(padded_docs, labels, epochs=100, verbose=0)
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('loss:', loss)
print('Accuracy:%f' % (accuracy * 100))
test = tf.keras.preprocessing.text.one_hot('good', 50)
padded_test = tf.keras.preprocessing.sequence.pad_sequences([test], maxlen=max_length, padding='post')
print(model.predict(padded_test))
