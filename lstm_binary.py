import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import larq
import quantize

if __name__ == "__main__":
    # Parameters
    batch_size = 128
    embedding_output_dims = 15
    max_length = 300
    num_words = 5000
    num_epochs = 5

    # Load dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    # Pad all sequences
    X_train_pad = pad_sequences(X_train, maxlen=max_length, value = 0.0) # 0.0 because it corresponds with <PAD>
    X_test_pad = pad_sequences(x_test, maxlen=max_length, value = 0.0) # 0.0 because it corresponds with <PAD>

    cv = quantize.make_clips(1, 1)

    # Build LSTM
    lstm = Sequential()
    lstm.add(Embedding(num_words, embedding_output_dims, input_length=max_length))
    lstm.add(quantize.QuantLSTM(10, 
        activation='hard_tanh', 
        recurrent_activation=quantize.hard_sigmoid, 
        kernel_quantizer=larq.quantizers.SteSign(clip_value=cv),
        kernel_constraint=larq.constraints.WeightClip(clip_value=1),
        recurrent_constraint=larq.constraints.WeightClip(clip_value=1)))
    lstm.add(larq.layers.QuantDense(units=1, kernel_quantizer=larq.quantizers.SteSign(clip_value=cv), kernel_constraint=larq.constraints.WeightClip(clip_value=1)))

    lstm.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=['accuracy'])

    lstm.fit(X_train_pad, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=1/5)

    results = results.evaluate(X_test_pad, y_test, verbose=False)
    print('Loss: ' + str(results[0]))
    print('Accuracy: ' + str(results[1]))
    print('Error: ' + str(1-results[1]))
