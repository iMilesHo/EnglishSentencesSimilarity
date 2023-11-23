import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk
import time

class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size=41699, d_model=128):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        # No need to explicitly define mean and normalization, can be done in forward

    def forward(self, x1, x2):
        # Assuming x1 and x2 are the input sequences for the two siamese branches
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # LSTM layer
        x1, _ = self.lstm(x1)
        x2, _ = self.lstm(x2)

        # Mean over sequence
        x1 = torch.mean(x1, dim=1)
        x2 = torch.mean(x2, dim=1)

        # Normalization (L2)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)

        return x1, x2

def triplet_loss_fn(v1, v2, margin=0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = torch.matmul(v1, v2.T)
    batch_size = scores.size(0)

    positive = torch.diag(scores)
    negative_without_positive = scores - 2.0 * torch.eye(batch_size, device=scores.device)
    closest_negative = negative_without_positive.max(dim=1).values

    negative_zero_on_duplicate = scores * (1.0 - torch.eye(batch_size, device=scores.device))
    mean_negative = torch.sum(negative_zero_on_duplicate, dim=1) / (batch_size - 1)

    triplet_loss1 = torch.maximum(torch.zeros_like(positive), margin - positive + closest_negative)
    triplet_loss2 = torch.maximum(torch.zeros_like(positive), margin - positive + mean_negative)

    triplet_loss = torch.mean(triplet_loss1 + triplet_loss2)
    return triplet_loss

def train_model(model, train_loader, val_loader, learning_rate=0.01, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        for q1_q2 in train_loader:
            q1, q2 = q1_q2[0].to(device), q1_q2[1].to(device)
            optimizer.zero_grad()
            v1, v2 = model(q1, q2)
            loss = triplet_loss_fn(v1, v2)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for q1_q2 in val_loader:
                q1, q2 = q1_q2[0].to(device), q1_q2[1].to(device)
                v1, v2 = model(q1, q2)
                val_loss += triplet_loss_fn(v1, v2).item()
        end_time = time.time()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Time: {end_time - start_time}")

def classify(test_Q1, test_Q2, y, threshold, model, vocab, batch_size=64):
    """Function to test the accuracy of the model in PyTorch.

    Args:
        test_Q1 (numpy.ndarray): Array of Q1 questions.
        test_Q2 (numpy.ndarray): Array of Q2 questions.
        y (numpy.ndarray): Array of actual target.
        threshold (float): Desired threshold.
        model (torch.nn.Module): The Siamese model.
        vocab (collections.defaultdict): The vocabulary used.
        batch_size (int, optional): Size of the batches. Defaults to 64.

    Returns:
        float: Accuracy of the model.
    """
    model.eval()  # Set the model to evaluation mode
    accuracy = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i in range(0, len(test_Q1), batch_size):
            q1 = torch.tensor(test_Q1[i:i + batch_size], dtype=torch.long).to(device)
            q2 = torch.tensor(test_Q2[i:i + batch_size], dtype=torch.long).to(device)
            y_test = y[i:i + batch_size]

            # Get model predictions
            v1, v2 = model(q1, q2)

            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(v1, v2)

            # Make predictions based on the threshold
            predictions = cos_sim > threshold

            # Update accuracy
            accuracy += torch.sum(predictions == torch.tensor(y_test, dtype=torch.bool)).item()

    # Compute overall accuracy
    accuracy = accuracy / len(test_Q1)

    return accuracy

def pad_sequences(sequences, max_len, pad_value=0):
    return np.array([seq + [pad_value] * (max_len - len(seq)) for seq in sequences])

def predict(question1, question2, threshold, model, vocab, verbose=False):
    """Function for predicting if two questions are duplicates in PyTorch.

    Args:
        question1 (str): First question.
        question2 (str): Second question.
        threshold (float): Desired threshold.
        model (torch.nn.Module): The Siamese model.
        vocab (collections.defaultdict): The vocabulary used.
        verbose (bool, optional): If the results should be printed out. Defaults to False.

    Returns:
        bool: True if the questions are duplicates, False otherwise.
    """
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize and numericalize questions
    q1 = [vocab.get(word, vocab['<UNK>']) for word in nltk.word_tokenize(question1)]
    q2 = [vocab.get(word, vocab['<UNK>']) for word in nltk.word_tokenize(question2)]

    # Pad questions
    max_len = max(len(q1), len(q2))
    q1 += [vocab['<PAD>']] * (max_len - len(q1))
    q2 += [vocab['<PAD>']] * (max_len - len(q2))

    # Convert to PyTorch tensors
    q1_tensor = torch.tensor([q1], dtype=torch.long).to(device)
    q2_tensor = torch.tensor([q2], dtype=torch.long).to(device)

    with torch.no_grad():
        v1, v2 = model(q1_tensor, q2_tensor)
        cos_sim = torch.nn.functional.cosine_similarity(v1, v2)
        res = cos_sim.item() > threshold

    if verbose:
        print("Q1 = ", question1)
        print("Q2 = ", question2)
        print("Cosine Similarity = ", cos_sim.item())
        print("Result = ", res)

    return res

if __name__ == '__main__':
    # 1. 实例化模型
    # 假设 vocab_size 为 10000，d_model 为 128
    # Create a Siamese Network instance
    model = SiameseNetwork()
    print(model)

    # 2. 准备测试数据
    # 创建一些随机数据作为模拟输入
    # 假设批量大小为 4，序列长度为 10
    test_input1 = torch.randint(0, 10000, (4, 10))
    test_input2 = torch.randint(0, 10000, (4, 10))

    # 3. 运行模型
    # 由于这是一个测试，我们不需要计算梯度
    with torch.no_grad():
        output1, output2 = model(test_input1, test_input2)

    # 4. 验证输出
    print("Output shape:", output1.shape, output2.shape)
    print("Output example:", output1[0], output2[0])

    # ------------------------------
    # TripletLoss 测试
    # 创建 TripletLoss 实例
    # 测试数据，转换为 torch.Tensor
    v1 = torch.tensor([[0.26726124, 0.53452248, 0.80178373], [-0.5178918, -0.57543534, -0.63297887]])
    v2 = torch.tensor([[0.26726124, 0.53452248, 0.80178373], [0.5178918, 0.57543534, 0.63297887]])

    # 计算损失
    loss = triplet_loss_fn(v1, v2, margin=0.25)

    print("Triplet Loss:", loss.item())

    # ------------------------------



