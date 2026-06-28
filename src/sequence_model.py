import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # CRITICAL CHANGE: We MUST switch to unidirectional (bidirectional=False).
        # A bidirectional LSTM cannot carry states forward sequentially in a live session
        # because the backward layer requires knowing the "future" end of the text.
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=False, batch_first=True
        )

        # Changed from hidden_dim * 2 to hidden_dim because it's now unidirectional
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence_tensor, hidden=None):
        embeds = self.embedding(sentence_tensor)

        # Pass the previous hidden state into the LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)

        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)

        # Return BOTH the scores and the updated memory state
        return tag_scores, hidden


class SequenceWrapper:
    def __init__(self, embedding_dim=32, hidden_dim=32):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Dictionaries to map strings to integers and vice versa
        self.word_to_ix = {}
        self.tag_to_ix = {}
        self.ix_to_tag = {}

        self.model = None
        self.is_trained = False

    def _prepare_sequence(self, seq, to_ix):
        """Helper to convert a list of words/tags into a PyTorch tensor of integers"""
        # If a word isn't in our vocab, we default to 0 (assuming 0 is 'UNK' or we just handle it)
        idxs = [to_ix.get(w, 0) for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def train(self, training_data):
        """
        training_data should be a list of tuples:
        [ (["put", "red", "block"], ["O", "B-T_COLOR", "B-T_SHAPE"]), ... ]
        """
        # 1. Build the vocabulary from the training data
        for sentence, tags in training_data:
            for word in sentence:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag[self.tag_to_ix[tag]] = tag

        # 2. Initialize the model NOW, because we finally know the vocab and tagset sizes
        self.model = LSTMModel(
            vocab_size=len(self.word_to_ix),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            tagset_size=len(self.tag_to_ix),
        )

        # 3. Setup Loss and Optimizer for Classification
        loss_function = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # 4. The Training Loop
        num_epochs = 50
        for epoch in range(num_epochs):
            total_loss = 0

            # Assume each element in training_data is a conversation session.
            # To simulate memory, we keep the hidden state alive across sentences.
            hidden = None

            for sentence, tags in training_data:
                # Step A: Clear old gradients
                self.model.zero_grad()

                # Step B: Prepare inputs
                sentence_in = self._prepare_sequence(
                    sentence, self.word_to_ix
                ).unsqueeze(0)
                targets = self._prepare_sequence(tags, self.tag_to_ix)

                # Step C: Run forward pass, passing the current hidden state
                tag_scores, hidden = self.model(sentence_in, hidden)

                # CRITICAL: Detach the hidden state from history.
                # This prevents PyTorch from trying to backpropagate all the way back
                # to the beginning of the universe, which causes an Out-Of-Memory error.
                hidden = (hidden[0].detach(), hidden[1].detach())

                # Step D: Calculate Loss
                loss = loss_function(
                    tag_scores.view(-1, len(self.tag_to_ix)), targets
                )

                # Step E: Backpropagate and update weights
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        self.is_trained = True

    # Add an extra method to reset memory when a new interaction begins
    def reset_session(self):
        """Call this whenever a brand new conversation or command chain starts."""
        self.current_hidden = None


    def predict_tags(self, sentence_tokens):
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet!")

        # Initialize current_hidden if it doesn't exist yet
        if not hasattr(self, "current_hidden"):
            self.current_hidden = None

        with torch.no_grad():
            inputs = self._prepare_sequence(
                sentence_tokens, self.word_to_ix
            ).unsqueeze(0)

            # Pass the persistent session memory into the model
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
            
        # We save everything needed to rebuild the exact same model later
        state = {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'word_to_ix': self.word_to_ix,
            'tag_to_ix': self.tag_to_ix,
            'ix_to_tag': self.ix_to_tag,
            'model_state_dict': self.model.state_dict()
        }
        
        # torch.save is PyTorch's built-in way to save dictionaries to disk
        torch.save(state, filepath)
        print(f"Sequence model successfully saved to {filepath}")

    def load(self, filepath):
        """Loads the saved state and reconstructs the PyTorch model."""
        # Load the dictionary back from the file
        state = torch.load(filepath)
        
        # 1. Restore dimensions and vocabularies
        self.embedding_dim = state['embedding_dim']
        self.hidden_dim = state['hidden_dim']
        self.word_to_ix = state['word_to_ix']
        self.tag_to_ix = state['tag_to_ix']
        self.ix_to_tag = state['ix_to_tag']
        
        # 2. Rebuild the empty model architecture
        self.model = LSTMModel(
            vocab_size=len(self.word_to_ix), 
            embedding_dim=self.embedding_dim, 
            hidden_dim=self.hidden_dim, 
            tagset_size=len(self.tag_to_ix)
        )
        
        # 3. Inject the saved weights into the empty model
        self.model.load_state_dict(state['model_state_dict'])
        
        # 4. Set the model to evaluation mode (turns off training-specific behaviors like Dropout)
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
