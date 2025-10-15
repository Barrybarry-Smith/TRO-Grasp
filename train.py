import os
import argparse
import torch
import wandb
from datetime import datetime
from omegaconf import OmegaConf
from dataset.CMapDataset import create_dataloader
from model.tro_graph import RobotGraph

def prepare_input(batch, device):

    robot_pc_initial = batch['robot_pc_initial']
    for batch_robot_pc_initial in robot_pc_initial:
        for link_name, link_pc in batch_robot_pc_initial.items():
            batch_robot_pc_initial[link_name] = link_pc.to(device)

    robot_pc_target = batch['robot_pc_target']
    for batch_robot_pc_target in robot_pc_target:
        for link_name, link_pc in batch_robot_pc_target.items():
            batch_robot_pc_target[link_name] = link_pc.to(device)

    batch['object_pc'] = batch['object_pc'].to(device)
    batch['object_pc_normal'] = batch['object_pc_normal'].to(device)
    batch['initial_q'] = [x.to(device) for x in batch['initial_q']]
    batch['target_q'] = [x.to(device) for x in batch['target_q']]
    batch['initial_se3'] = [x.to(device) for x in batch['initial_se3']]
    batch['target_se3'] = [x.to(device) for x in batch['target_se3']]
    batch['initial_vec'] = [x.to(device) for x in batch['initial_vec']]
    batch['target_vec'] = [x.to(device) for x in batch['target_vec']]
    
    return batch


def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = config.train.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(
        project=config.train.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    print("Building dataloader...")
    dataloader = create_dataloader(config.dataset, is_train=True)
    print("Building model...")
    model = RobotGraph(**config.model).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params {total_params}")
    wandb.config.update({"total_params": total_params})

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.lr_step,
        gamma=config.train.lr_gamma
    )

    if config.train.resume_from:
        ckpt = torch.load(config.train.resume_from)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from {config.train.resume_from} at epoch {start_epoch}")
    else:
        start_epoch = 0

    num_epochs = config.train.epochs
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_id, batch in enumerate(dataloader):

            batch = prepare_input(batch, device)
            loss_dict = model(batch)
            loss = loss_dict['loss_total']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            log_data = {k: v.item() for k, v in loss_dict.items()}
            log_data.update({"lr": scheduler.get_last_lr()[0]})
            wandb.log(log_data)

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch_avg_loss": avg_epoch_loss})
        
        if (epoch + 1) % config.train.save_interval == 0:
            os.makedirs(os.path.join(save_dir, "ckpt"), exist_ok=True)
            ckpt_path = os.path.join(save_dir, "ckpt", f"{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
            }, ckpt_path)
            wandb.save(ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train(config)
