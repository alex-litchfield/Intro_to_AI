import torch
import torch.nn as nn
from torch.nn import functional as F
from models import Attention

class Transformer_Block(nn.Module):
    """
    This class builds the basic transformer block.
    """
    def __init__(self, n_embd, block_size):
        super().__init__()

        self.attn_block = Attention(n_embd, block_size)
        self.norm_1 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Step 1: Call the attention block
        attn_output = self.attn_block(x)
        
        # Step 2: Sum the output of attention and the input, then normalize
        x = self.norm_1(x + attn_output)
        
        # Step 3: Apply a linear layer followed by ReLU activation
        linear_output = F.relu(self.linear_1(x))
        
        # Step 4: Sum the output of linear layer and normalized input, then normalize
        x = self.norm_2(x + linear_output)

        return x


class Character_GPT(nn.Module):
    def __init__(self, block_size, n_embd, n_layer, vocab_size):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)  # Embedding layer

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [Transformer_Block(n_embd, block_size) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(n_embd)  # Normalization Layer
        self.output_layer = nn.Linear(n_embd, vocab_size, bias=False)

    def get_loss(self, input, target):
        output = self(input)
        return F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), ignore_index=-1)

    def forward(self, input):
        """
        Takes in an input sequence and outputs character probabilities.
        """
        # Step 1: Apply embedding layer
        x = self.embed(input)  # Shape: (batch, seq_len, n_embd)

        # Step 2: Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Step 3: Apply normalization
        x = self.norm(x)

        # Step 4: Apply output linear layer (without activation)
        output = self.output_layer(x)  # Shape: (batch, seq_len, vocab_size)

        return output

    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generates text based on a given prompt.
        """
        for _ in range(max_new_tokens):
            # Ensure we respect the block size constraint
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward pass through the model to get logits
            logits = self(idx_cond)  # Expected shape: (batch_size, seq_len, vocab_size)

            # Debugging print
            print("Shape of logits before selecting last step:", logits.shape)  # Debugging output

            # Fix: Select only the last token’s logits correctly
            logits = logits[:, -1, :].squeeze(0)  # Ensure shape: (1, vocab_size)

            # Apply softmax to convert logits into probabilities
            probs = F.softmax(logits, dim=-1)  # Correct shape: (1, vocab_size)

            # Debugging print statements
            print("Shape of logits after selecting last step:", logits.shape)  # Should be (1, 65)
            print("Shape of probs after softmax:", probs.shape)  # Should be (1, 65)

            # Sample the next index from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # Shape: (1, 1)

            # Fix: Ensure `idx_next` and `idx` have the same number of dimensions
            if idx.dim() > idx_next.dim():
                idx_next = idx_next.unsqueeze(0)  # Ensure it matches `idx`'s batch size

            # Append the sampled token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # Shape: (1, sequence_length + 1)

        return idx







