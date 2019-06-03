import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

raw_text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

raw_text_size = len(raw_text)

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}  # 96

# CBOW  [context, target]
cbow_data = []
for i in range(CONTEXT_SIZE, raw_text_size - CONTEXT_SIZE):
    context = []
    for j in range(CONTEXT_SIZE, 0, -1):
        context.append(raw_text[i - j])
    for j in range(1, CONTEXT_SIZE + 1, +1):
        context.append(raw_text[i + j])
    target = raw_text[i]
    cbow_data.append((context, target))


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * 2 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        return out


losses = []
model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for name, params in model.named_parameters():
    print(name, ':', params.size())

for epoch in range(40):
    total_loss = 0
    for context, target in cbow_data:
        context_idxs = torch.LongTensor([word_to_ix[w] for w in context])
        target = torch.LongTensor([word_to_ix[target]])

        optimizer.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)

plt.plot(losses)
plt.show()
