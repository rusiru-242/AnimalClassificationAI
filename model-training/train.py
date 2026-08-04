from utils.dataset_loader import load_datasets

train_dataset, validation_dataset, class_names = load_datasets()

print("\nClasses")

print(class_names)

print("\nTraining Dataset")

print(train_dataset)

print("\nValidation Dataset")

print(validation_dataset)