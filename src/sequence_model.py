import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Unidirectional LSTM is used to allow sequential forward state propagation in live sessions.
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=False, batch_first=True
        )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence_tensor, hidden=None):
        embeds = self.embedding(sentence_tensor)
        lstm_out, hidden = self.lstm(embeds, hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores, hidden


class SequenceWrapper:
    def __init__(self, embedding_dim=32, hidden_dim=32):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_to_ix = {}
        self.tag_to_ix = {}
        self.ix_to_tag = {}

        self.model = None
        self.is_trained = False

    def _prepare_sequence(self, seq, to_ix):
        """Helper to convert a list of words/tags into a PyTorch tensor of integers"""
        # Out-of-vocabulary words default to index 0.
        idxs = [to_ix.get(w, 0) for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def train(self, training_data):
        """
        training_data should be a list of tuples:
        [ (["put", "red", "block"], ["O", "B-T_COLOR", "B-T_SHAPE"]), ... ]
        """
        # Build vocabulary from training data
        for sentence, tags in training_data:
            for word in sentence:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag[self.tag_to_ix[tag]] = tag

        # Initialize model once vocabulary and tagset sizes are known
        self.model = LSTMModel(
            vocab_size=len(self.word_to_ix),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            tagset_size=len(self.tag_to_ix),
        )

        # Setup loss and optimizer
        loss_function = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            total_loss = 0
            hidden = None

            for sentence, tags in training_data:
                self.model.zero_grad()

                sentence_in = self._prepare_sequence(
                    sentence, self.word_to_ix
                ).unsqueeze(0)
                targets = self._prepare_sequence(tags, self.tag_to_ix)

                tag_scores, hidden = self.model(sentence_in, hidden)

                # Detach the hidden state to truncate backpropagation through time.
                hidden = (hidden[0].detach(), hidden[1].detach())

                loss = loss_function(
                    tag_scores.view(-1, len(self.tag_to_ix)), targets
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        self.is_trained = True

    def reset_session(self):
        """Call this whenever a brand new conversation or command chain starts."""
        self.current_hidden = None

    def predict_tags(self, sentence_tokens):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet!")

        if not hasattr(self, "current_hidden"):
            self.current_hidden = None

        with torch.no_grad():
            inputs = self._prepare_sequence(
                sentence_tokens, self.word_to_ix
            ).unsqueeze(0)

            tag_scores, self.current_hidden = self.model(inputs, self.current_hidden)

            _, predicted_indices = torch.max(tag_scores, dim=2)
            predicted_tags = [
                self.ix_to_tag[idx.item()] for idx in predicted_indices[0]
            ]

            return predicted_tags
    
    def save(self, filepath):
        """Saves the architecture dimensions, vocabularies, and model weights."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
            
        state = {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'word_to_ix': self.word_to_ix,
            'tag_to_ix': self.tag_to_ix,
            'ix_to_tag': self.ix_to_tag,
            'model_state_dict': self.model.state_dict()
        }
        
        torch.save(state, filepath)
        print(f"Sequence model successfully saved to {filepath}")

    def load(self, filepath):
        """Loads the saved state and reconstructs the PyTorch model."""
        state = torch.load(filepath)
        
        self.embedding_dim = state['embedding_dim']
        self.hidden_dim = state['hidden_dim']
        self.word_to_ix = state['word_to_ix']
        self.tag_to_ix = state['tag_to_ix']
        self.ix_to_tag = state['ix_to_tag']
        
        self.model = LSTMModel(
            vocab_size=len(self.word_to_ix), 
            embedding_dim=self.embedding_dim, 
            hidden_dim=self.hidden_dim, 
            tagset_size=len(self.tag_to_ix)
        )
        
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval() 
        self.is_trained = True
        
        print(f"Sequence model successfully loaded from {filepath}")


if __name__ == "__main__":
    training_data = [
        (
            ["put", "the", "red", "block", "inside", "the", "wooden", "box"],
            [
                "O",
                "O",
                "B-T_COLOR",
                "B-T_SHAPE",
                "B-REL",
                "O",
                "B-D_MAT",
                "B-D_SHAPE",
            ],
        ),
        (
            ["inspect", "the", "large", "blue", "sphere"],
            ["O", "O", "B-T_SIZE", "B-T_COLOR", "B-T_SHAPE"],
        ),
    ]

    wrapper = SequenceWrapper()
    wrapper.train(training_data)

    test_sentence = ["inspect", "the", "red", "sphere"]
    predictions = wrapper.predict_tags(test_sentence)

    print("\nTest Sentence:", test_sentence)
    print("Predictions:  ", predictions)
