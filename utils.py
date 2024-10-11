# Function to save a checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# Model evaluation code
@torch.no_grad()
def calculate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', test_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(iter(loader.load_dataset()))
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out






    