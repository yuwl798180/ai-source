import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(1)


def example():
    # args: input_size, hidden_size, bidirectional=False, num_layers=1, dropout=0, batch_first=False
    lstm = nn.LSTM(3, 3)

    # input_size: shape (seq_len, batch, input_size)
    inputs = [torch.randn(1, 3) for _ in range(5)]
    inputs = torch.cat(inputs).view(5, 1, -1)  # (5, 3) => (5, 1, 3)

    # h0: (num_layers * num_directions, batch, hidden_size)  default: 0
    # c0: (num_layers * num_directions, batch, hidden_size)  default: 0
    hidden = (
        torch.randn(1, 1, 3),  # initial hidden state
        torch.randn(1, 1, 3))  # initial cell state

    # Inputs: input, (h_0, c_0)
    # Outputs: output, (h_n, c_n)
    out, hidden = lstm(inputs, hidden)
    # out shape: (seq_len, batch, num_directions * hidden_size)
    # h_n shape: (num_layers * num_directions, batch, hidden_size)

    print('out:', out, '\n', 'hidden:', hidden)


EMBEDDING_DIM = 8
HIDDEN_DIM = 12

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_id = {}
tag_to_id = {"DET": 0, "NN": 1, "V": 2}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)
vocab_size = len(word_to_id)
tagset_size = len(tag_to_id)


def prepare_senquence(seq, to_id):
    ids = [to_id[w] for w in seq]
    return torch.LongTensor(ids)


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # print(embeds.size())  # sen_len * embedding_dim  5*8
        lstm_out, _ = self.lstm(embeds.view(
            len(sentence), 1, -1))  # input: sen_len,batch,input_size
        # print(lstm_out.size())  # sen_len * batch * hidden_dim 5*1*12
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # print(tag_space.size()) # sen_len * tagset_size 5 * 3
        return tag_space


model = LSTMTagger(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, tagset_size)
loss_funcion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
with torch.no_grad():
    inputs = prepare_senquence(training_data[0][0], word_to_id)  # 1*5
    tag_space = model(inputs)
    tags = F.softmax(tag_space, dim=1)
    # print(tags)  # 5*3
    _, tags_idx = torch.max(tags, dim=1)
    print(tags_idx)

losses = []
# loop over the dataset multiple times
for epoch in range(200):
    running_loss = 0.0
    for sentence, tags in training_data:
        sentence_input = prepare_senquence(sentence, word_to_id)
        targets = prepare_senquence(tags, tag_to_id)

        optimizer.zero_grad()

        tag_space = model(sentence_input)
        loss = loss_funcion(tag_space, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # print('Loss: {}'.format(running_loss))
    losses.append(running_loss)

plt.plot(losses)
plt.show()
print('Finished Training!')

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_senquence(training_data[0][0], word_to_id)  # 1*5
    tag_space = model(inputs)
    tags = F.softmax(tag_space, dim=1)
    # print(tags)  # 5*3
    _, tags_idx = torch.max(tags, dim=1)
    print(tags_idx)  # [0, 1, ,2 ,0 ,1]
