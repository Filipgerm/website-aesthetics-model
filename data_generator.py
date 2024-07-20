import numpy as np
import tensorflow as tf

# # Custom data generator function
# def rating_data_generator(X, y, batch_size):
#     dataset_size = X.shape[0]
#     while True:
#         indices = np.arange(dataset_size)
#         np.random.shuffle(indices)
#         for start_idx in range(0, dataset_size, batch_size):
#             end_idx = min(start_idx + batch_size, dataset_size)
#             batch_indices = indices[start_idx:end_idx]
#             yield X[batch_indices] / 255.0, y[batch_indices]

def rating_data_generator(X, y, batch_size):
    """
    Custom data generator function for CNN model.
    
    Parameters:
    X (numpy.ndarray): Input data.
    y (numpy.ndarray): Target labels.
    batch_size (int): Size of each batch.
    
    Yields:
    tuple: Tuple containing a batch of input data and corresponding labels.
    """
    # Validate inputs
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"
    
    dataset_size = X.shape[0]
    indices = np.arange(dataset_size)
    
    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices] / 255.0, y[batch_indices]

            

# Custom generator for comparison data
def comparison_data_generator(image_pairs, labels, batch_size):
    """
    Custom data generator function for comparison task.
    
    Parameters:
    image_pairs (list of tuples): List of tuples where each tuple contains two images (image_a, image_b).
    labels (numpy.ndarray): Array of labels corresponding to each pair.
    batch_size (int): Size of each batch.
    
    Yields:
    tuple: Tuple containing a batch of pairs of input data and corresponding labels.
    """
    num_samples = len(image_pairs)
    while True:
        # Shuffle the data at the start of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            image_a_batch = []
            image_b_batch = []
            label_batch = []

            for i in batch_indices:
                image_a, image_b = image_pairs[i]

                # Resizing already done
                # image_a = read_and_process_images(image_path1)    
                # image_b = read_and_process_images(image_path2)

                # Normalize the images
                image_a = image_a.astype(np.float32) / 255.0
                image_b = image_b.astype(np.float32) / 255.0

                label = labels[i]

                # Debugging prints for each image
                # print(f"Image A shape: {image_a.shape}, Image B shape: {image_b.shape}")

                image_a_batch.append(image_a)
                image_b_batch.append(image_b)
                label_batch.append(label)

                # # Debugging prints
                # print("Batch shapes:")
                # print("image_a_batch:", np.array(image_a_batch).shape)
                # print("image_b_batch:", np.array(image_b_batch).shape)
                # print("label_batch:", np.array(label_batch).shape)

            yield (np.array(image_a_batch), np.array(image_b_batch)), np.array(label_batch)


# Combined generator
def combined_data_generator(rating_gen, comparison_gen, batch_size):
    """
    Combines two generators into one, yielding batches of data for both tasks.
    
    Parameters:
    rating_gen (generator): Generator for the rating task.
    comparison_gen (generator): Generator for the comparison task.
    batch_size (int): Size of each batch.
    
    Yields:
    tuple: Tuple containing combined input data and corresponding labels.
    """

    # Convert generators to iterators
    rating_iter = iter(rating_gen)
    comparison_iter = iter(comparison_gen)

    while True:
        # Get the next batch of data from each generator
        rating_data, rating_labels = next(rating_iter)
        comparison_data, comparison_labels = next(comparison_iter)


        # # Check if either batch is smaller than the specified batch size
        # if rating_data.shape[0] < batch_size or comparison_data[0].shape[0] < batch_size:
        #     continue  # Drop this batch and move to the next one

        #         # Ensure both generators yield batches of the same size
        # min_batch_size = min(rating_data.shape[0], comparison_data[0].shape[0])

        # # Trim batches to the minimum batch size
        # rating_data = rating_data[:min_batch_size]
        # rating_labels = rating_labels[:min_batch_size]
        # comparison_data = [cd[:min_batch_size] for cd in comparison_data]
        # comparison_labels = comparison_labels[:min_batch_size]


    


        # if min_batch_size < batch_size:
        #     # Calculate the number of padding samples needed
        #     rating_pad_count = batch_size - min_batch_size
        #     comparison_pad_count = batch_size - min_batch_size

        #     # Pad rating data and labels
        #     rating_data = np.pad(rating_data, ((0, rating_pad_count), (0, 0), (0, 0), (0, 0)), mode='wrap')
        #     rating_labels = np.pad(rating_labels, (0, rating_pad_count), mode='wrap')

        #     # Pad comparison data and labels
        #     comparison_data[0] = np.pad(comparison_data[0], ((0, comparison_pad_count), (0, 0), (0, 0), (0, 0)), mode='wrap')
        #     comparison_data[1] = np.pad(comparison_data[1], ((0, comparison_pad_count), (0, 0), (0, 0), (0, 0)), mode='wrap')
        #     comparison_labels = np.pad(comparison_labels, (0, comparison_pad_count), mode='wrap')

        #         # Debugging prints
        # print("Rating data shape:", rating_data.shape)
        # print("Comparison data shapes:", [cd.shape for cd in comparison_data])
        # print("Rating labels shape:", rating_labels.shape)
        # print("Comparison labels shape:", comparison_labels.shape)

        # # Ensure both generators yield batches of the same size
        # if rating_data.shape[0] != comparison_data[0].shape[0]:
        #     print("Batch size mismatch detected!")
        #     print("Rating data batch size:", rating_data.shape[0])
        #     print("Comparison data batch size:", comparison_data[0].shape[0])

        #  # Ensure both generators yield batches of the same size
        # assert rating_data.shape[0] == comparison_data[0].shape[0], "Batch sizes do not match!"

        # Combine the data into the format required by the joint model
        combined_inputs = (rating_data, comparison_data[0], comparison_data[1])
        combined_labels = (rating_labels, comparison_labels)

        yield combined_inputs, combined_labels