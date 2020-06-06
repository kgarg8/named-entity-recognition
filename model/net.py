"""Defines the neural networks, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, params):
        """
        We define a recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:
        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()

        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are padding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.
        Returns:
            out: (Variable) dimension batch_size * seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.
        """

        s = self.embedding(s)       # dim: batch_size x seq_len x embedding_dim
        s, _ = self.lstm(s)
        s = s.contiguous()          # required often before view
        s = s.view(-1, s.shape[2])  # dim: batch_size*seq_len x lstm_hidden_dim
        s = self.fc(s)
        # softmax on all tokens (batch_size -> #sentences, seq_len -> #tokens_per_sentence, s-> all tokens)
        return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):
    """Compute the cross entropy loss over outputs from the model and labels for all tokens

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
            or -1 in case it is a PADding token.
    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch
    """

    # reshape to shape of batch_size*seq_len
    labels = labels.view(-1)

    # since padding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask
    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens


def accuracy(outputs, labels):
    """Compute accuracy for all tokens excluding Padding terms"""

    labels = labels.ravel()  # flattened array
    mask = (labels >= 0)
    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(np.sum(mask))


# maintain all metrics required in this dictionary - these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # add more metrics if required for each token type
}
