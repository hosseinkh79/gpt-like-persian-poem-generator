import torch
from engine import train
from model import PersinaPoemGPT
from config import Config
from tokenizer.bpe import SimpleBPE
from dataloader import create_dataloaders

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate Config if needed
    config = Config()
    
    model = PersinaPoemGPT(config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    tokenizer = SimpleBPE.load("data/tokenizer.json")
    print(f'vocab_size: {tokenizer.vocab_size}')

    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        block_size=config.block_size,
        batch_size=4
    )

    train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device
    )

if __name__ == "__main__":
    main()