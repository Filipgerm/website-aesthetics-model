import os
import csv
from sklearn.model_selection import train_test_split

# Data folder paths
data_folder = '../../Calista/website-aesthetics-datasets-master/rating-based-dataset/preprocess/'
train_data_path = data_folder + 'train_means_list.csv'
test_data_path = data_folder + 'test_list.csv'
images_path = data_folder + 'resized'

# Comparison data paths
comparison_dataset = '../../Calista/website-aesthetics-datasets-master/comparison-based-dataset/'
comparison_images_path = comparison_dataset + 'resized_images'
comparison_data_path = comparison_dataset + 'data/comparisons.csv'

# Function to get image IDs from CSV
def get_image_ids_from_csv(file_path):
    image_ids = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip header
        for row in csv_reader:
            image_path = row[0]
            image_id = os.path.basename(image_path).split('.')[0]  # Extract ID from file name
            image_ids.append(image_path)
    return image_ids

# Get the image paths from train and test CSV files
train_image_paths = get_image_ids_from_csv(train_data_path)
test_image_paths = get_image_ids_from_csv(test_data_path)

# Combine train and test image paths
all_image_paths = train_image_paths + test_image_paths

# print("sorted(all_image_paths))", sorted(all_image_paths))

# Create a dictionary to map the original paths to their new comparison IDs
path_to_comparison_id = {path: i for i, path in enumerate(sorted(all_image_paths))}

# Print the correspondence for verification
for original_path, comparison_id in path_to_comparison_id.items():
    print(f"{original_path} -> {comparison_id}")

# Extract the IDs for train and test sets based on the new mapping
train_image_ids = [path_to_comparison_id[path] for path in train_image_paths]
test_image_ids = [path_to_comparison_id[path] for path in test_image_paths]

print(f"Number of training images: {len(train_image_ids)}")
print(f"Number of testing images: {len(test_image_ids)}")
print(f"Total number of images: {len(path_to_comparison_id)}")

# Save the mapped image paths and their IDs to a new CSV file
mapped_image_paths_csv = 'mapped_image_paths.csv'
with open(mapped_image_paths_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['original_path', 'comparison_id'])
    for original_path, comparison_id in path_to_comparison_id.items():
        csv_writer.writerow([original_path, comparison_id])

def get_comparison_data(comparison_csv_path, images_path, valid_image_ids, file_extension='.png'):
    image_pairs = []
    labels = []
    with open(comparison_csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip header
        for row in csv_reader:
            im1 = int(row[1])
            im2 = int(row[2])
            w1 = int(row[3])
            w2 = int(row[4])

            if im1 in valid_image_ids and im2 in valid_image_ids:
                image_path1 = os.path.join(images_path, f"{im1}{file_extension}")
                image_path2 = os.path.join(images_path, f"{im2}{file_extension}")

                if w1 != w2:  # Skip ties
                    label = 1 if w1 > w2 else -1
                    image_pairs.append((image_path1, image_path2))
                    labels.append(label)
    return image_pairs, labels

# Combine valid image IDs for the whole dataset
valid_image_ids = set(train_image_ids + test_image_ids)

image_pairs, pairs_labels = get_comparison_data(comparison_data_path, comparison_images_path, valid_image_ids)
print(f"Filtered image pairs: {len(image_pairs)}")


# Save the filtered image pairs and labels to new CSV files for training and testing
train_pairs_csv = 'train_image_pairs.csv'
test_pairs_csv = 'test_image_pairs.csv'

def save_image_pairs_to_csv(image_pairs, labels, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path1', 'image_path2', 'label'])
        for (img1, img2), label in zip(image_pairs, labels):
            csv_writer.writerow([img1, img2, label])

def filter_pairs(image_pairs, labels, valid_image_ids):
    filtered_pairs = []
    filtered_labels = []
    for (img1, img2), label in zip(image_pairs, labels):
        img1_id = int(os.path.basename(img1).split('.')[0])
        img2_id = int(os.path.basename(img2).split('.')[0])
        if img1_id in valid_image_ids and img2_id in valid_image_ids:
            filtered_pairs.append((img1, img2))
            filtered_labels.append(label)
    return filtered_pairs, filtered_labels

train_image_pairs, train_labels = filter_pairs(image_pairs, pairs_labels, set(train_image_ids))
test_image_pairs, test_labels = filter_pairs(image_pairs, pairs_labels, set(test_image_ids))

save_image_pairs_to_csv(train_image_pairs, train_labels, train_pairs_csv)
save_image_pairs_to_csv(test_image_pairs, test_labels, test_pairs_csv)

print(f"Training pairs: {len(train_image_pairs)}")
print(f"Testing pairs: {len(test_image_pairs)}")
