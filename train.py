import torch
from torch import nn
import random
from tqdm import tqdm
import sys


# hyperparameters
SEED = 666
BLOCK_SIZE = 256        
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
N_LAYER = 6
EPOCHS = 5000
EMBED_DIM = 384
HEAD_SIZE = 16
N_HEAD = 4
LR = 3e-4
DROPOUT = 0.2
FILENAME = "data/tolstoi_l_n__voina_i_mir.txt"


def print_model_info(model):
    n_params = 0
    for params in model.parameters():
        n_params += params.numel()
    print(f'GPT 0.5 model has {n_params} parameters')


def create_train_model(filename: str = FILENAME,
                       seed: int = SEED,
                       block_size: int = BLOCK_SIZE,        
                       device: str = DEVICE,
                       batch_size: str = BATCH_SIZE,
                       n_layer: int  = N_LAYER,
                       epochs: int = EPOCHS,
                       embed_dim: int = EMBED_DIM,
                       n_head: int = N_HEAD,
                       lr: float = LR,
                       dropout: float = DROPOUT,
                       model_name: str = "models/gpt0_5.pth",
                       train: bool = False):
    """Create model with initialized arguments.

        Args:
            filename (str): name of the file that model was trained on
            seed (int): seed
            block_size (int): size of block        
            device (str): 'cpu' or 'cuda'
            batch_size (str): batch size
            n_layer (int) : number of decoder blocks
            epochs (int): epochs
            embed_dim (int): embedding dimension
            n_head (int): amount of head in multihead attention
            lr (float): learning rate
            dropout (float): dropout coefficient
        Returns:
            (model, loss_fn, optimizer)"""
    # get data
    with open(filename) as file:
        data = file.read()

    ind2char = {ind: char for ind, char in enumerate(sorted(list(set(data))))}
    char2ind = {char: ind for ind, char in enumerate(sorted(list(set(data))))}

    DATA_LEN = len(data)
    VOCAB_LEN = len(ind2char)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # functions
    encode = lambda x: torch.LongTensor([char2ind[i] for i in x])
    decode = lambda x: "".join([ind2char[i.item()] for i in x])

    def get_batch():
        """get random (X, y) sample from data"""

        start = random.randint(0, DATA_LEN - block_size * batch_size * 2)

        X = torch.stack([torch.LongTensor(encode(data[i:i + block_size])) for i in range(start, start + batch_size * block_size + 1, block_size)])
        y = torch.stack([torch.LongTensor(encode(data[i + 1:i + block_size + 1])) for i in range(start, start + batch_size * block_size + 2, block_size)])

        return X.to(device), y.to(device)


    # model
    class Head(nn.Module):
        """one head of QKV self-attention"""
        def __init__(self, head_size=embed_dim):
            super().__init__()

            self.query = nn.Linear(embed_dim, head_size, bias=False)
            self.key = nn.Linear(embed_dim, head_size, bias=False)
            self.value = nn.Linear(embed_dim, head_size, bias=False)

            self.dropout = nn.Dropout(dropout)

        
        def forward(self, x):
            B, T, C = x.shape    
            tril = torch.tril(torch.ones((T, T))).to(device)

            q = self.query(x)
            k = self.key(x)
            v = self.value(x)

            w = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, T)
            w = w.masked_fill(tril == 0, float('-inf')).softmax(-1)
            w = self.dropout(w)
            out = w @ v

            return out


    class FeedForward(nn.Module):
        def __init__(self, embed_dim=embed_dim):
            super().__init__()

            self.fdfw = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.fdfw(x)

    class MultiHeadAttention(nn.Module):
        def __init__(self, n_heads=n_head, head_size=embed_dim):
            super().__init__()

            self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
            self.proj = nn.Linear(n_heads * head_size , n_heads * head_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([head(x) for head in self.heads], -1)
            out = self.proj(out)
            out = self.dropout(out)
            return out


    class Block(nn.Module):
        def __init__(self, n_heads=n_head, embed_dim=embed_dim):
            super().__init__()

            self.heads = MultiHeadAttention(head_size=embed_dim // n_heads)

            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)

            self.fwfd = FeedForward(embed_dim)

        def forward(self, x):
            out = x + self.heads(self.ln1(x))
            out = out + self.fwfd(self.ln2(out))
            return out


    class LangModel(nn.Module):
        def __init__(self, embed_dim=embed_dim, n_layer=n_layer):
            super().__init__()

            self.token_embed = nn.Embedding(VOCAB_LEN, embed_dim)
            self.position_embed = nn.Embedding(block_size, embed_dim)

            self.blocks = nn.Sequential(
                *[Block() for _ in range(n_layer)],
                nn.LayerNorm(embed_dim)
            )
            
            self.heads = MultiHeadAttention(head_size=embed_dim // n_head)
            self.fwfd = FeedForward(embed_dim)
            self.classifier = nn.Linear(embed_dim, VOCAB_LEN)


        def forward(self, x):
            tokens_embed = self.token_embed(x) # (B, T, C)
            pos_embed = self.position_embed(torch.arange(0, x.shape[1], device=device)) # (B, T, C)
            
            out = tokens_embed + pos_embed
            out = self.blocks(out)
            out = self.classifier(out)

            return out
        
        def generate(self, idx=torch.zeros((1, 1), dtype=torch.int64).to(DEVICE), len=block_size):
            for _ in range(len):
                probs = torch.softmax(self(idx[:, -block_size:]), 2)[:, -1, :]
                next_idx = torch.multinomial(probs, 1)
                idx = torch.cat((idx, next_idx), 1)
                print(decode(idx.data[0])[-1], end='')
            print(decode(idx.data[0])[-1], end='')
            return idx

        
    model = LangModel(embed_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    print_model_info(model)

    if train:
        print('Model starts training...')
        model.train()
        for i in tqdm(range(1, EPOCHS + 1)):
            X, y = get_batch()
            y_pred = model(X)
            B, T, C = y_pred.shape
            loss = loss_fn(y_pred.reshape(B * T, C), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f'{i} epoch, loss - {loss.item()}')
        
        torch.save(model.state_dict(), model_name)


    return model, loss_fn, optimizer

def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    argv = sys.argv 
    train = False
    if len(argv) == 2:
        filename = argv[1]
        train = True
    else:
        filename = FILENAME

    model, _, _ = create_train_model(filename=filename, train=train)

if __name__ == '__main__':
    main()