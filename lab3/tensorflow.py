import numpy as np
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255 # Ši eilutė konvertuoja duomenis į float32 tipo skaičius ir normalizuoja juos.
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1) # papildomą matmenį, kad būtų išlaikytas tinklas, kuris turi numatytąjį paveikslėlių formos 
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Function to plot MNIST images
def plot_images(original_images, augmented_images, labels, num_images=3):
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        # Original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title("Original: {}".format(int(labels[i])))
        plt.axis("off")
        
        # Augmented image
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(augmented_images[i].numpy().reshape(28, 28), cmap='gray')
        plt.title("Augmented")
        plt.axis("off")
    plt.savefig('augmented_images.png')  # Save augmented images plot
    plt.show()

"""
## Data Augmentation
"""

data_augmentation_layers = [
    layers.RandomRotation(0.005),
    layers.RandomContrast(0.005),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

x_train_augmented = data_augmentation(x_train)

# Plot the images
plot_images(x_train, x_train_augmented, y_train)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape), # sukuria modelio struktura
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),  
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),  
        layers.Dropout(0.5),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),  
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(), #pavercia is 2d i 1d
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"), # sumeta visus neuronus i viena 
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 15

# Define the checkpoint callback to save the model with the lowest validation loss
checkpoint_filepath = 'best_model.keras'
last_model_checkpoint_path = 'last_model.keras'

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True, verbose=1)

# Define a callback to save the last model after training
last_model_checkpoint = ModelCheckpoint(filepath=last_model_checkpoint_path,
                                        save_weights_only=False,
                                        verbose=0)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train_augmented, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[model_checkpoint_callback, last_model_checkpoint])

"""
## Evaluate the trained model
"""
best_model = keras.models.load_model(checkpoint_filepath)
last_model = keras.models.load_model(last_model_checkpoint_path)

score = model.evaluate(x_test, y_test, verbose=0)
best_score = best_model.evaluate(x_test, y_test, verbose=0)
print("Last model: ")
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("-------------------------------")
print("Best model: ")
print("Test loss:", best_score[0])
print("Test accuracy:", best_score[1])
plt.figure(figsize=(10, 10))

def plot_predictions(model, images, labels, num_images=3):
    plt.figure(figsize=(15, 6))
    num_total_images = images.shape[0]
    random_indices = np.random.choice(num_total_images, size=num_images, replace=False)
    predictions = model.predict(images[random_indices])
    for i, idx in enumerate(random_indices):
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap='gray_r')
        
        # Get top 3 classes and their probabilities
        top_classes = np.argsort(-predictions[i])[:3]
        top_probs = predictions[i][top_classes]
        
        # Format title text
        title_text = '\n'.join([f'Class: {cls}, Probability: {prob:.2%}' for cls, prob in zip(top_classes, top_probs)])
        
        plt.title(title_text, color='#017653')
        plt.axis("off")
    plt.savefig('predictions_plot.png')  # Save predictions plot
    plt.show()

# Plot predictions for the best model
plot_predictions(best_model, x_test, y_test)

def load_and_evaluate_models():
    # Load the saved best model
    best_checkpoint_filepath = 'best_model.keras'
    best_model = keras.models.load_model(best_checkpoint_filepath)

    # Load the saved last model
    last_checkpoint_filepath = 'last_model.keras'
    last_model = keras.models.load_model(last_checkpoint_filepath)

    # Load the test data
    (x_test, y_test) = keras.datasets.mnist.load_data()[1]
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Evaluate the best model
    best_score = best_model.evaluate(x_test, y_test, verbose=0)

    # Evaluate the last model
    last_score = last_model.evaluate(x_test, y_test, verbose=0)

    results = f"Best model test results:\nTest loss: {best_score[0]}\nTest accuracy: {best_score[1]}\n\n"
    results += f"Last model test results:\nTest loss: {last_score[0]}\nTest accuracy: {last_score[1]}"

    return results

# Call the function to load and evaluate the best and last models
results = load_and_evaluate_models()

# Save the results to a text file
with open('model_results.txt', 'w') as file:
    file.write(results)

print("Model results saved to model_results.txt")
