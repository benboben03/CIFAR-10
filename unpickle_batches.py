import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Unpickle all dataset batches to retrieve trainig data, training labels, testing data, and testing labels.
def load_cifar():
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = 'test_batch'
    train_data = []
    train_labels = []
    for batch in train_batches:
        data_dict = unpickle("cifar_10_batches/" + batch)
        train_data.append(data_dict[b'data'])
        train_labels += data_dict[b'labels']
    test_dict = unpickle("cifar_10_batches/" + test_batch)
    test_data = test_dict[b'data']
    test_labels = test_dict[b'labels']
    return np.concatenate(train_data), np.array(train_labels), test_data, np.array(test_labels)


'''
- Divide pixel values by 255 to get pixel values to range between [0, 1]
- Original CIFAR data is stored as flattened array in which each row represents an image, so
  reshaping turns each image into (32x32x3), 3 represents rgb.
- Transpose rearranges the structure of the matrix by making the channel dimension data to come last
  aka height x width x channels
'''
def preprocess_data(train_data, test_data):
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    return train_data, test_data

def write_data_to_text(train_data, train_labels, test_data, test_labels, train_file, test_file):
    # Write training data and labels to text file
    with open(train_file, 'w') as f:
        f.write("Training Data:\n")
        for data in train_data:
            f.write(','.join(map(str, data)) + '\n')
        f.write("\nTraining Labels:\n")
        f.write(','.join(map(str, train_labels)) + '\n')
    
    # Write testing data and labels to text file
    with open(test_file, 'w') as f:
        f.write("Testing Data:\n")
        for data in test_data:
            f.write(','.join(map(str, data)) + '\n')
        f.write("\nTesting Labels:\n")
        f.write(','.join(map(str, test_labels)) + '\n')

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_cifar()
    train_data, test_data = preprocess_data(train_data, test_data)

    # train_file = 'train_data.txt'
    # test_file = 'test_data.txt'
    # write_data_to_text(train_data, train_labels, test_data, test_labels, train_file, test_file)
