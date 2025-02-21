import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import accuracy_score
import ssl
import matplotlib.pyplot as plt
import numpy as np
import time

tf.compat.v1.enable_eager_execution()
# Handle SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Data Preparation and Preprocessing
train_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomCrop(height=32, width=32, seed=1337),
    layers.Rescaling(1./255),
    layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])
])

# Load CIFAR-10 dataset
(train_images_cifar10, train_labels_cifar10), (test_images_cifar10, test_labels_cifar10) = tf.keras.datasets.cifar10.load_data()

# Normalize the data using Rescaling
train_images_cifar10, test_images_cifar10 = train_images_cifar10 / 255.0, test_images_cifar10 / 255.0

# Flatten labels
train_labels_cifar10 = train_labels_cifar10.flatten()
test_labels_cifar10 = test_labels_cifar10.flatten()

# 2. Model Initialization (ResNet and VGG)
def initialize_model(model_name, num_classes=10):
    if model_name == "resnet":
        base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation='softmax')
        ])
    elif model_name == "vgg":
        base_model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(num_classes, activation='softmax')
        ])
    return model

# Choose model (ResNet or VGG)
model_name = "resnet"  # Change to "vgg" for VGG model
num_classes = 10  # CIFAR-10 has 10 classes
model = initialize_model(model_name, num_classes)

# 3. Optimizer and Learning Rate Scheduler
# Optimizer setup
optimizer = SGD(learning_rate=1e-3, momentum=0.9)

# Learning rate scheduler
class CyclicLRScheduler(Callback):
    def __init__(self, base_lr, max_lr, step_size):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def on_epoch_end(self, epoch, logs=None):
        cycle = epoch // (2 * self.step_size)
        x = abs(epoch / self.step_size - 2 * cycle - 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        
        # Set the learning rate directly using model's optimizer
        self.model.optimizer.learning_rate.assign(lr)
        print(f"Updated learning rate (CLR): {lr}")

# CLRPlus Scheduler
class CLRPlusScheduler(Callback):
    def __init__(self, base_lr, max_lr, step_size, adjust_on="loss", adjustment_factor=0.1):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.iteration = 0
        self.adjust_on = adjust_on
        self.adjustment_factor = adjustment_factor
        self.prev_loss = None

    def on_epoch_end(self, epoch, logs=None):
        # Loss trend adjustment
        if self.adjust_on == "loss" and logs.get('loss') is not None:
            loss = logs['loss']
            if self.prev_loss is not None and loss > self.prev_loss:
                self.max_lr *= (1 - self.adjustment_factor)  # Shrink learning rate range
                self.base_lr *= (1 - self.adjustment_factor)
            self.prev_loss = loss
        
        # Calculate cyclical learning rate
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        
        # Set the learning rate directly using model's optimizer
        self.model.optimizer.learning_rate.assign(lr)
        print(f"Updated learning rate (CLRPlus): {lr}")

        self.iteration += 1

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Early stopping callback

# Compile the model
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Training the Model
def train_model_with_scheduler(model, train_images, train_labels, scheduler, num_epochs=10, batch_size=128):
    start_time = time.time()  # Start timer to track training time
    history = model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, 
                        callbacks=[scheduler, early_stopping], validation_split=0.1)
    training_time = time.time() - start_time  # End timer and calculate training time
    return history, training_time

# 5. Evaluate the Model
def evaluate_model(model, test_images, test_labels):
    predictions = model.predict(test_images)
    predicted_labels = predictions.argmax(axis=-1)

    accuracy = accuracy_score(test_labels, predicted_labels)

    return accuracy

# Plotting function for performance metrics
def plot_metrics(history, scheduler_name, training_time):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Add training time to the title
    plt.suptitle(f'{scheduler_name} - Training Time: {training_time:.2f}s', fontsize=14)

    # Save the plot to the specified file
    plt.savefig(f"{scheduler_name}.png", dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to {scheduler_name}.png")
    plt.close() 

# 6. Train and Evaluate on CIFAR-10 using CyclicLRScheduler
print("Training with CyclicLRScheduler")
cyclic_lr = CyclicLRScheduler(base_lr=1e-6, max_lr=1e-2, step_size=200)
history_cyclic, training_time_cyclic = train_model_with_scheduler(model, train_images_cifar10, train_labels_cifar10, cyclic_lr, num_epochs=10, batch_size=128)

# Evaluate and plot metrics for CyclicLRScheduler
accuracy_cyclic = evaluate_model(model, test_images_cifar10, test_labels_cifar10)
print(f"Accuracy (CyclicLRScheduler): {accuracy_cyclic * 100:.2f}%")
plot_metrics(history_cyclic, "CyclicLRScheduler", training_time_cyclic)

# 7. Train and Evaluate on CIFAR-10 using CLRPlusScheduler
print("Training with CLRPlusScheduler")
clr_plus_lr = CLRPlusScheduler(base_lr=1e-6, max_lr=1e-2, step_size=200, adjust_on="loss")
history_clr_plus, training_time_clr_plus = train_model_with_scheduler(model, train_images_cifar10, train_labels_cifar10, clr_plus_lr, num_epochs=10, batch_size=128)

# Evaluate and plot metrics for CLRPlus
accuracy_clr_plus = evaluate_model(model, test_images_cifar10, test_labels_cifar10)
print(f"Accuracy (CLRPlusScheduler): {accuracy_clr_plus * 100:.2f}%")
plot_metrics(history_clr_plus, "CLRPlusScheduler", training_time_clr_plus)
