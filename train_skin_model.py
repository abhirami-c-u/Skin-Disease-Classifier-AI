import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# 1. Load metadata
df = pd.read_csv('skin_disease_data/HAM10000_metadata.csv')

# 2. Find image path
def find_image_path(image_id):
    part1 = os.path.join("skin_disease_data", "HAM10000_images_part_1", image_id + ".jpg")
    part2 = os.path.join("skin_disease_data", "HAM10000_images_part_2", image_id + ".jpg")
    return part1 if os.path.exists(part1) else (part2 if os.path.exists(part2) else None)

df['image_path'] = df['image_id'].apply(find_image_path)
df = df[df['image_path'].notnull()]

# 3. Keep selected 5 classes
selected_labels = ['akiec', 'bcc', 'bkl', 'mel', 'df']
df = df[df['dx'].isin(selected_labels)]

# 4. Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

# 5. Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = datagen.flow_from_dataframe(train_df, x_col='image_path', y_col='dx',
                                        target_size=(224, 224), class_mode='categorical', batch_size=32)

val_gen = datagen.flow_from_dataframe(val_df, x_col='image_path', y_col='dx',
                                      target_size=(224, 224), class_mode='categorical', batch_size=32)

# 6. Compute class weights
sorted_classes = np.array(sorted(train_df['dx'].unique()))
class_weights_array = compute_class_weight(class_weight='balanced', classes=sorted_classes, y=train_df['dx'])
class_weights = {train_gen.class_indices[cls]: weight for cls, weight in zip(sorted_classes, class_weights_array)}

# 7. Model building
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(len(sorted_classes), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 8. Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('skin_classifier/best_model.keras', save_best_only=True, monitor='val_loss')
]

# 9. Train model (initial phase)
model.fit(train_gen, validation_data=val_gen, epochs=15, class_weight=class_weights, callbacks=callbacks)

# 10. Fine-tune by unfreezing top layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weights, callbacks=callbacks)

# 11. Save final model
model.save('skin_classifier/final_skin_model.keras')
print("Final model saved to skin_classifier/final_skin_model.keras")

# 12. Evaluation
val_gen.reset()
preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
labels = list(train_gen.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
