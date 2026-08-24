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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01)
        
    loss_fn = torch.nn.CrossEntropyLoss()

    tokenizer = SimpleBPE.load("data/tokenizer.json")
    print(f'vocab_size: {tokenizer.vocab_size}')

    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        block_size=config.block_size,
        batch_size=16
    )

    results = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config.epochs,
        device=device
    )
    
    plot_loss_curves(results=results)

    text = "در هوای تو"

    # Encode and move to device
    input_ids = tokenizer.encode(text)
    input_idx = torch.tensor(input_ids).unsqueeze(0).long().to(device)

    # Generate (model is already on device)
    out = model.generate(input_idx, max_new_tokens=100)

    # Move output back to CPU for decoding (optional but recommended)
    out_cpu = out.cpu()
    print(f'output_tokens: {out_cpu[0]}')
    print(tokenizer.decode(out_cpu[0].tolist()))

if __name__ == "__main__":
    main()