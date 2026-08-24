import torch
from engine import train
from model import PersianPoemGPT
from config import Config
from tokenizer.bpe import SimpleBPE
from dataloader import create_dataloaders
from utils import plot_loss_curves

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate Config if needed
    config = Config()
    
    model = PersianPoemGPT(config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    tokenizer = SimpleBPE.load("data/tokenizer.json")
    print(f'vocab_size: {tokenizer.vocab_size}')

    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        block_size=config.block_size,
        batch_size=4
    )

    results = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=1,
        device=device
    )
    
    plot_loss_curves(results=results)

    text = "در هوای تو"
    # input_ids = tokenizer.encode(text)
    # print(input_ids)
    input_idx = torch.tensor(tokenizer.encode(text)).unsqueeze(0).long()
    out = model.generate(input_idx, max_new_tokens=100)
    print(f'output_tokens: {out[0]}')
    print(tokenizer.decode(out[0].tolist()))

if __name__ == "__main__":
    main()