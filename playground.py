from learning_rate_scheduler import GPTLearningRateScheduler
import Dataloader
from SmolLM import SmolLM
import torch 
import torch.nn as nn 
from preprocessing import read_file_path



model =SmolLM().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

num_epochs = 10
max_iter = int((len(dataset) / batch_size) * num_epochs)
lr_scheduler = GPTLearningRateScheduler(max_lr=1e-5, min_lr=0.0, warm_up_iters=2000, max_iters=max_iter)
learning_rate = lr_scheduler.get_lr(0)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
wandb.init(project='smol LLM Design Experiments')

global_iter = 0 

for epoch in range(num_epochs):
  model.train()
  for x, y in train_iter:
    #Evaluate the loss, calculate gradient, update weight
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    lr = lr_scheduler.get_lr(current_iter=global_iter)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    if global_iter % evaluation_intervals == 0 or global_iter == max_iter - 1:
      model.eval()
      losses = calculate_loss()
      checkpoint = {
                'iter': global_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses}
      save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{global_iter}.pth.tar")
      print(f"step {global_iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Learning Rate: {lr:.6f}")
      wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "learning_rate":lr})

      model.train()

    global_iter += 1