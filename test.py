import os
import jax
import json
import time
import tqdm
import torch
import argparse
import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf

from dataset.CMapDataset import create_dataloader
from model.tro_graph import RobotGraph
from utils.hand_model import create_hand_model
from utils.pyroki_ik import PyrokiRetarget
from utils.optimization import *
from validation.validate_utils import validate_isaac

def prepare_input(batch, device):

    batch['object_pc'] = batch['object_pc'].to(device)
    batch['initial_q'] = [x.to(device) for x in batch['initial_q']]
    
    return batch

def test(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building dataloader...")
    dataloader = create_dataloader(config.dataset, is_train=False)
    print("Building model...")
    model = RobotGraph(**config.model).to(device)
    
    state_dict = torch.load(config.test.ckpt)["model_state"]
    model.load_state_dict(state_dict, strict=False)

    with open("data/data_urdf/robot/urdf_assets_meta.json", 'r') as f:
        robot_urdf_meta = json.load(f)
    
    #### Compile Jax for Pyroki ####
    robot_name = config.test.embodiment
    hand = create_hand_model(robot_name, device)
    urdf_path = robot_urdf_meta["urdf_path"][robot_name]
    target_links = list(hand.links_pc.keys())
    ik_solver = PyrokiRetarget(urdf_path, target_links)
    batch_retarget = jax.jit(ik_solver.solve_retarget)
    ################################

    success_dict = {}
    diversity_dict = {}
    vis_info = []
    batch_size = config.dataset.batch_size
    model.eval()
    total_inference_time = 0
    total_grasp_num = 0
    warmed_up = False

    for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):

        transform_dict = {}
        data_count = 0
        predict_q_list = []
        initial_q_list = []      # Unconditioned
        initial_se3_list = []    # Conditioned
        object_pc_list = []
        transform_list = []
        while data_count != batch_size:
            split_num = min(batch_size - data_count, config.test.split_batch_size)
            initial_q = batch["initial_q"][data_count : data_count + split_num].to(device)
            initial_se3 = batch['initial_se3'][data_count : data_count + split_num].to(device)
            object_pc = batch['object_pc'][data_count : data_count + split_num].to(device)
            robot_links_pc = batch['robot_links_pc'][data_count : data_count + split_num]
            split_batch = {
                'robot_name': batch['robot_name'],
                'object_name': batch['object_name'],
                'initial_q': initial_q,
                'initial_se3': initial_se3,
                'object_pc': object_pc,
                'robot_links_pc': robot_links_pc
            }
            data_count += split_num

            time_start = time.time()
            with torch.no_grad():
                all_diffuse_step_poses_dict = model.inference(split_batch)
  
            ## IK process with pyroki
            clean_robot_pose = all_diffuse_step_poses_dict[0]
            optim_transform = process_transform(hand.pk_chain, clean_robot_pose)
            initial_q_jnp = jnp.array(initial_q.cpu().numpy())
            target_pos_list = [optim_transform[name] for name in target_links]
            target_pos = torch.stack(target_pos_list, dim=1)
    
            target_pos_jnp = jnp.array(target_pos.detach().cpu().numpy())
            predict_q_jnp = batch_retarget(
                initial_q=initial_q_jnp,
                target_pos=target_pos_jnp
            )
            jax.block_until_ready(predict_q_jnp)
            time_end = time.time()
            
            if warmed_up:
                total_inference_time += (time_end - time_start)
                total_grasp_num += split_num
            else:
                warmed_up = True

            predict_q = torch.from_numpy(np.array(predict_q_jnp)).to(device=device, dtype=initial_q.dtype)
            initial_q_list.append(initial_q)
            initial_se3_list.append(initial_se3)
            predict_q_list.append(predict_q)
            object_pc_list.append(object_pc)
            for diffuse_step, pred_robot_pose in all_diffuse_step_poses_dict.items():
                if diffuse_step not in transform_dict:
                    transform_dict[diffuse_step] = []
                transform_dict[diffuse_step].append(pred_robot_pose)
        
        # Simulation, isaac subprocess
        all_predict_q = torch.cat(predict_q_list, dim=0)
        
        success, isaac_q = validate_isaac(
            batch['robot_name'], 
            batch["object_name"], 
            all_predict_q, 
            gpu=config.test.gpu
        )
        success_dict[batch["object_name"]] = success

        # Diversity
        success_q = all_predict_q[success]
        diversity_dict[batch["object_name"]] = success_q

        for diffuse_step, transform_list in transform_dict.items():
            transform_batch = {}
            for transform in transform_list:
                for k, v in transform.items():
                    transform_batch[k] = v if k not in transform_batch else torch.cat((transform_batch[k], v), dim=0)
            transform_dict[diffuse_step] = transform_batch

        vis_info.append({
            'robot_name': batch['robot_name'],
            'object_name': batch['object_name'],
            'initial_q': torch.cat(initial_q_list, dim=0),
            'initial_se3': torch.cat(initial_se3_list, dim=0),
            'predict_q': torch.cat(predict_q_list, dim=0),
            'object_pc': torch.cat(object_pc_list, dim=0),
            'predict_transform': transform_dict,
            'success': success,
            'isaac_q': isaac_q
        })

    os.makedirs(config.test.save_dir, exist_ok=True)
    torch.save(
        vis_info,
        os.path.join(config.test.save_dir, "vis.pt")
    )

    output_path = os.path.join(config.test.save_dir, "res.txt")
    with open(output_path, "w") as f:
        total_success = 0
        total_sum = 0
        for obj, obj_res in success_dict.items():
            line = f"{obj}: {obj_res.sum() / len(obj_res)}\n"
            print(line, end="")
            f.write(line)
            total_success += obj_res.sum()
            total_sum += len(obj_res)

        line = f"Total success rate: {total_success / total_sum}.\n"
        print(line, end="")
        f.write(line)

        all_success_q = torch.cat(list(diversity_dict.values()), dim=0)
        diversity_std = torch.std(all_success_q, dim=0).mean()
        line = f"Total diversity: {diversity_std}\n"
        print(line, end="")
        f.write(line)

        line = f"Grasp generation time: {total_inference_time / total_grasp_num} s.\n"
        print(line, end="")
        f.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    test(config)
