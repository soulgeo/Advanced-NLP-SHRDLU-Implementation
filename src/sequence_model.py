import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence_tensor):
        # 1. Convert integer indexes into dense vector embeddings
        embeds = self.embedding(sentence_tensor)
        # 2. Pass through LSTM (We ignore the hidden states output here)
        lstm_out, _ = self.lstm(embeds)
        # 3. Project to tag space
        tag_space = self.hidden2tag(lstm_out)
        # 4. Convert to log probabilities
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores


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

            for sentence, tags in training_data:
                # Step A: Clear old gradients
                self.model.zero_grad()

                # Step B: Prepare inputs and targets as tensors. Add batch dimension: shape (1, seq_len)
                sentence_in = self._prepare_sequence(
                    sentence, self.word_to_ix
                ).unsqueeze(0)
                targets = self._prepare_sequence(tags, self.tag_to_ix)

                # Step C: Run forward pass
                tag_scores = self.model(sentence_in)

                # Step D: Calculate Loss
                # PyTorch loss functions expect the predictions to be 2D (N, C) and targets to be 1D (N)
                # So we view/reshape tag_scores from (1, seq_len, num_tags) to (seq_len, num_tags)
                loss = loss_function(
                    tag_scores.view(-1, len(self.tag_to_ix)), targets
                )

                # Step E: Backpropagate and update weights
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(training_data):.4f}'
                )

        self.is_trained = True

    def predict_tags(self, sentence_tokens):
        """Takes a list of words and returns a list of predicted tags."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained yet!")

        with torch.no_grad():  # Don't track gradients during prediction
            # Convert words to tensor
            inputs = self._prepare_sequence(
                sentence_tokens, self.word_to_ix
            ).unsqueeze(0)

            # Get predictions
            tag_scores = self.model(inputs)

            # Find the index of the highest probability tag for each word
            # tag_scores shape is (1, seq_len, num_tags). max(dim=2) gets the best tag per word
            _, predicted_indices = torch.max(tag_scores, dim=2)

            # Convert integers back to string tags
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
