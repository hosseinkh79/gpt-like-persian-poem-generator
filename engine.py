import torch
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F


# Train and validation
def one_step_train(model, train_dataloader, loss_fn, optimizer, device):
    model = model.to(device)

    model.train()
    train_loss = 0

    for batch, (X, y) in enumerate(train_dataloader):
        # print(f'X: {X.shape}')        
        
        X, y = X.to(device), y.to(device)

        # print(f'X: {X.shape}')
        # print(f'y: {y.shape}')

        logits = model(X)
        # print(f'logits: {logits.shape}')
        
        B, T, C = logits.shape
        
        logits = logits.view(B*T, C)
        # print(f'logits: {logits.shape}')

        y = y.view(B*T)
        # print(f'y: {y.shape}')


        loss = loss_fn(logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss = train_loss/len(train_dataloader)

    return train_loss


def one_step_val(model, val_dataloader, loss_fn, device):
    model = model.to(device)

    model.eval()
    val_loss = 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X)
            # print(f'logits: {logits.shape}')
            
            B, T, C = logits.shape
            
            logits = logits.view(B*T, C)
            y = y.view(B*T)

            loss = loss_fn(logits, y)

            val_loss += loss.item()

    val_loss = val_loss/len(val_dataloader)

    return val_loss


def train(model,
          train_dataloader,
          val_dataloader,
          loss_fn,
          optimizer,
          device,
          epochs=1):

    results = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(epochs):

        train_loss = one_step_train(model,
                                               train_dataloader,
                                               loss_fn, optimizer,
                                               device)

        val_loss = one_step_val(model,
                                         val_dataloader,
                                         loss_fn,
                                         device)

        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
        )

    return results