"""Train the image classifier using a CIFAKE-style train/{FAKE,REAL} dataset."""
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", help="Dataset root containing train/FAKE, train/REAL and test/...")
    parser.add_argument("--output", type=Path, default=Path("models/AI_Media_Guard_Image_Final.keras"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=12000, help="Balanced random subset size; use 0 for all training images")
    parser.add_argument("--max-test-samples", type=int, default=5000, help="Random subset size; use 0 for all test images")
    args = parser.parse_args()
    if args.dataset is None:
        candidates = [Path("archive (1)"), Path("dataset"), Path("data")]
        args.dataset = next((p for p in candidates if (p / "train" / "FAKE").is_dir()), candidates[0])
    train_dir, test_dir = args.dataset / "train", args.dataset / "test"
    for folder in (train_dir / "FAKE", train_dir / "REAL", test_dir / "FAKE", test_dir / "REAL"):
        if not folder.is_dir():
            raise SystemExit(f"Expected dataset folder not found: {folder}")

    image_size, batch_size = (128, 128), 64
    dataset_options = dict(directory=train_dir, image_size=image_size, batch_size=batch_size, label_mode="binary", seed=42, validation_split=.1)
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **dataset_options)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", shuffle=True, **dataset_options)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=image_size, batch_size=batch_size, label_mode="binary", shuffle=args.max_test_samples > 0, seed=43)
    if train_ds.class_names != ["FAKE", "REAL"]:
        raise SystemExit(f"Expected class order ['FAKE', 'REAL']; found {train_ds.class_names}")
    if args.max_train_samples > 0:
        train_ds = train_ds.take((args.max_train_samples + batch_size - 1) // batch_size)
        val_ds = val_ds.take(max(1, (args.max_train_samples // 10 + batch_size - 1) // batch_size))
    if args.max_test_samples > 0:
        test_ds = test_ds.take((args.max_test_samples + batch_size - 1) // batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    augment = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(.05), layers.RandomZoom(.1), layers.RandomContrast(.1)])
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*image_size, 3))
    base.trainable = False
    inputs = layers.Input((*image_size, 3))
    x = augment(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    loss, accuracy, auc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {loss:.4f} | accuracy: {accuracy:.4f} | AUC: {auc:.4f}")
    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
