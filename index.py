import multiprocessing
from TransformerEncoder import TransformerEncoder
from PositionalEmbedding import PositionalEmbedding
from tensorflow import keras
from tensorflow.keras import layers

# divide into train set, validation set and test set
batch_size = 32
train_ds = keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory('aclImdb/val', batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory('aclImdb/test', batch_size=batch_size)

# Text pre-processing
max_length = 600
max_tokens = 2000

text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_length,
)

text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

num_parallel_calls = multiprocessing.cpu_count()
int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=num_parallel_calls)
int_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=num_parallel_calls)
int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=num_parallel_calls)

# sequence model
vocab_size = 20000
sequence_length = 600
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype='int64')
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras", save_best_only=True)]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
model = keras.models.load_model("full_transformer_encoder.keras", custom_objects={"TransformerEncoder": TransformerEncoder, "PositionalEmbedding": PositionalEmbedding})
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
