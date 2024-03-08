from data_preprocessing import split_data, tokens_to_sentence
from Siamese_Model import SiameseNetwork, train_model, classify_and_confusion_matrix, pad_sequences, predict
import numpy as np
import pandas as pd
import nltk
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import pickle

# this part is for training, including data preprocessing, model training, and model saving

file_path = "myResource/questions.csv"

# 1. Load the data
data = pd.read_csv(file_path)
N_train = 300000
N_test = 10 * 1024
data_train = data[:N_train]
data_test = data[N_train:N_train + N_test]
del data

# 2. Extracting duplicate question pairs
td_index = (data_train['is_duplicate'] == 1).to_numpy()
td_index = [i for i, x in enumerate(td_index) if x]

Q1_train_words = np.array(data_train['question1'][td_index])
Q2_train_words = np.array(data_train['question2'][td_index])

Q1_test_words = np.array(data_test['question1'])
Q2_test_words = np.array(data_test['question2'])
y_test = np.array(data_test['is_duplicate'])

# 3. Building the vocabulary
Q1_train = np.empty_like(Q1_train_words)
Q2_train = np.empty_like(Q2_train_words)

Q1_test = np.empty_like(Q1_test_words)
Q2_test = np.empty_like(Q2_test_words)

vocab = defaultdict(lambda: 0)
vocab['<PAD>'] = 1

for idx in range(len(Q1_train_words)):
    Q1_train[idx] = nltk.word_tokenize(Q1_train_words[idx])
    Q2_train[idx] = nltk.word_tokenize(Q2_train_words[idx])
    q = Q1_train[idx] + Q2_train[idx]
    for word in q:
        if word not in vocab:
            vocab[word] = len(vocab) + 1
# Save the vocab dictionary to a file
# Convert defaultdict to a regular dictionary
vocab_dict = dict(vocab)
with open('./myResource/vocab.pkl', 'wb') as f:
    pickle.dump(vocab_dict, f)

# 4. Tokenizing and numerically encoding the questions
for idx in range(len(Q1_train)):
    Q1_train[idx] = [vocab[word] for word in Q1_train[idx]]
    Q2_train[idx] = [vocab[word] for word in Q2_train[idx]]

for idx in range(len(Q1_test_words)):
    Q1_test[idx] = [vocab[word] for word in nltk.word_tokenize(Q1_test_words[idx])]
    Q2_test[idx] = [vocab[word] for word in nltk.word_tokenize(Q2_test_words[idx])]

# 5. Padding the sequences to a common length
max_length = max(max(len(q) for q in Q1_train), max(len(q) for q in Q2_train))

Q1_train = [q + [vocab['<PAD>']] * (max_length - len(q)) for q in Q1_train]
Q2_train = [q + [vocab['<PAD>']] * (max_length - len(q)) for q in Q2_train]

# 6. Splitting the train data into training and validation sets
train_Q1, train_Q2, val_Q1, val_Q2 = split_data(Q1_train, Q2_train)
print("Number of training pairs:", len(train_Q1))
print("the first training question in Q1", train_Q1[0])
print("the first training question in Q2", train_Q2[0])

# 7. Converting the data into PyTorch tensors
train_Q1_tensor = torch.tensor(train_Q1, dtype=torch.long)
train_Q2_tensor = torch.tensor(train_Q2, dtype=torch.long)
# Check shapes and types
print("Shape of train_Q1_tensor:", train_Q1_tensor.shape)
print("Shape of train_Q2_tensor:", train_Q2_tensor.shape)

val_Q1_tensor = torch.tensor(val_Q1, dtype=torch.long)
val_Q2_tensor = torch.tensor(val_Q2, dtype=torch.long)
# Check shapes and types
print("Shape of val_Q1_tensor:", val_Q1_tensor.shape)
print("Shape of val_Q2_tensor:", val_Q2_tensor.shape)

# 8. Creating the DataLoader
train_dataset = TensorDataset(train_Q1_tensor, train_Q2_tensor)
val_dataset = TensorDataset(val_Q1_tensor, val_Q2_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Check the first batch in the train_dataset

print("\ncheck the first batch in the train_dataset")
# Creating the inverse vocabulary for converting the tokens back to words
inverse_vocab = {v: k for k, v in vocab.items()}
# Try to get one batch
try:
    # train_features is a list of two tensors for Q1 and Q2
    train_features = next(iter(train_loader))
    print("DataLoader works! train data shapes:", [d.shape for d in train_features])

    # Example usage
    tokenized_example = np.array(train_features[0][0])
    original_Q1_sentence = tokens_to_sentence(tokenized_example, inverse_vocab)
    tokenized_example = np.array(train_features[1][0])
    original_Q2_sentence = tokens_to_sentence(tokenized_example, inverse_vocab)
    print(original_Q1_sentence)
    print(original_Q2_sentence)
except Exception as e:
    print("Error with DataLoader:", e)


# check the first batch in the val_dataset
print("\ncheck the first batch in the val_dataset")
try:
    # val_features is a list of two tensors for Q1 and Q2
    val_features = next(iter(val_loader))
    print("DataLoader works! val data shapes:", [d.shape for d in val_features])

    # Example usage
    tokenized_example = np.array(val_features[0][0])
    original_Q1_sentence = tokens_to_sentence(tokenized_example, inverse_vocab)
    tokenized_example = np.array(val_features[1][0])
    original_Q2_sentence = tokens_to_sentence(tokenized_example, inverse_vocab)
    print(original_Q1_sentence)
    print(original_Q2_sentence)
except Exception as e:
    print("Error with DataLoader:", e)

# 9. Instantiate the model
print("\nInstantiate the model, train the model, and save the model")
model = SiameseNetwork(vocab_size=len(vocab), d_model=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device:", device)
print("Model:", model)
print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Train the model
print("hyperparameters: learning_rate=0.01, epochs=10")
train_model(model, train_loader, val_loader, learning_rate=0.01, epochs=10)
print("Done training!")

# Save the entire model
torch.save(model, './model/model_entire.pth')

# 10. Test the model on the test set
# Determine the maximum length
max_length = max(max(len(seq) for seq in Q1_test), max(len(seq) for seq in Q2_test))

# Pad the sequences
test_Q1_padded = pad_sequences(Q1_test, max_length, vocab['<PAD>'])
test_Q2_padded = pad_sequences(Q2_test, max_length, vocab['<PAD>'])

# Example usage:
accuracy, cm = classify_and_confusion_matrix(test_Q1_padded, test_Q2_padded, y_test, 0.7, model, vocab, batch_size=64)
print("Accuracy:", accuracy)

if __name__ == '__main__':
    pass

