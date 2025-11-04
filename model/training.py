"""
General training script for training SAEs using the trainer
(Originally from https://github.com/saprmarks/dictionary_learning/blob/2d586e417cd30473e1c608146df47eb5767e2527/training.py)
"""
import json
import os
import time
import torch as t
from dictionary import AutoEncoder
from trainer import SAETrainer
from config import my_config

def train_run(data, my_config, trainer: SAETrainer):
    # Training loop
    step = 0
    start_time = time.time()
    for epoch in range(my_config['n_epoch']):
        for batch_data in data:
            act = batch_data['repr'].to(my_config["device"])
            
            # Logging metric
            if my_config['log_steps'] is not None and step % my_config['log_steps'] == 0:
                log = {}
                log['epoch'] = epoch
                log['step'] = step
                # record time for each log period
                log['time'] = time.time() - start_time
                start_time = time.time()

                with t.no_grad():
                    act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)
                    # Aggregate logging metrics
                    log.update(losslog)

                    # Check for NaN loss
                    if losslog["loss"] != losslog["loss"]:
                        print("Oh no, NaN loss!!")
                        breakpoint()

                    # Calculate sparsity metrics
                    n_nonzero_per_example = (f != 0).float().sum(dim=-1)
                    l0 = n_nonzero_per_example.mean().item()
                    log["l0"] = l0

                    # Calculate variance explained
                    total_variance = t.var(act, dim=0).sum()
                    residual_variance = t.var(act - act_hat, dim=0).sum()
                    frac_variance_explained = 1 - residual_variance / total_variance
                    log["frac_variance_explained"] = frac_variance_explained.item()

                    # Log activation statistics
                    log["act_mean"] = act.mean().item()
                    log["act_std"] = act.std().item()
                    log["reconstruction_mean"] = act_hat.mean().item()
                    log["reconstruction_std"] = act_hat.std(dim=1).mean().item()
                    log["batch_tokens"] = act.shape[0]

                    # write the log to file
                    with open(os.path.join(my_config['save_dir'], f"training_log.json"), "a") as f:
                        json.dump(log, f)
                        f.write('\n')

            # Save checkpoints
            if my_config['save_steps'] is not None and ((step % my_config['save_steps'] == 0)):
                chk_path = os.path.join(my_config['save_dir'], "checkpoints", f"step_{step}.pt")
                if not os.path.exists(chk_path):
                    t.save(trainer.ae.state_dict(),chk_path)

            trainer.update(step, act)
            step += 1