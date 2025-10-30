"""
Implements the standard SAE training scheme.
(Originally from https://github.com/saprmarks/dictionary_learning/blob/2d586e417cd30473e1c608146df47eb5767e2527/trainers/standard.py)
"""

from collections import namedtuple
import os
import torch as t

from dictionary import AutoEncoder
from config import my_config

class SAETrainer:
    """
    Generic class for implementing SAE training algorithms.

    Base class that provides common functionality for SAE training implementations.
    Subclasses should implement the update method to define specific training behavior.

    Args:
        seed: Random seed for reproducibility
    """

    def __init__(self, seed=None):
        self.seed = seed
        self.logging_parameters = []

    def update(self,
               step: int,  # index of step in training
               activations: t.Tensor,  # shape [batch_size, d_submodule]
               ):
        """
        Update the model based on current step and activations.

        Args:
            step: Current training step number
            activations: Batch of input activations to process
        """
        pass  # implemented by subclasses

    def get_logging_parameters(self):
        """
        Collect all registered logging parameters from the trainer.

        Returns:
            Dictionary mapping parameter names to their current values
        """
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                stats[param] = getattr(self, param)
            else:
                print(f"Warning: {param} not found in {self}")
        return stats

    @property
    def config(self):
        """Basic configuration dictionary for the trainer."""
        return {
            'wandb_name': 'trainer',
        }


class ConstrainedAdam(t.optim.Adam):
    """
    Adam optimizer variant that maintains unit norm constraints on specified parameters.

    Implements a modified Adam optimizer that projects gradients and renormalizes
    parameters after each update to maintain unit norm constraints.

    Args:
        params: All parameters to optimize
        constrained_params: Parameters that should maintain unit norm
        lr: Learning rate
    """

    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        """
        Performs a single optimization step with norm constraints.

        1. Projects gradients for constrained parameters
        2. Performs standard Adam update
        3. Renormalizes constrained parameters
        """
        with t.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0,
                                                  keepdim=True) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)


class StandardTrainer(SAETrainer):
    """
    Standard SAE training implementation with L1 sparsity and neuron resampling.

    Implements training with:
    - L1 sparsity penalty
    - Learning rate warmup
    - Dead neuron detection and resampling
    - L1 penalty annealing

    Args:
        dict_class: Autoencoder class to use (default: AutoEncoder)
        activation_dim: Dimension of input activations
        dict_size: Size of the learned dictionary
        lr: Learning rate
        l1_penalty: Final L1 penalty coefficient
        warmup_steps: Steps for learning rate warmup
        l1_annealing_pct: Fraction of training to anneal L1 penalty
        steps: Total training steps
        resample_steps: Frequency of neuron resampling
        seed: Random seed
        device: Computing device
        layer: Model layer being processed
        plm_name: Protein language model name
        wandb_name: W&B run name
        submodule_name: Name of processed submodule
    """

    def __init__(self, my_config, dict_class=AutoEncoder):
        super().__init__(my_config['seed'])

        # Set random seeds
        if my_config['seed'] is not None:
            t.manual_seed(my_config['seed'])
            t.cuda.manual_seed_all(my_config['seed'])

        # Initialize autoencoder
        self.ae = dict_class(my_config['activation_dim'], my_config['dict_size'], tied=my_config['tied'])

        # Training parameters
        self.dict_size = my_config['dict_size']
        self.l1_penalty = my_config['l1_penalty']
        self.softmin = t.nn.Softmin(dim=0)

        self.smooth_penalty = my_config['smooth_penalty']
        self.save_dir = my_config['save_dir']
        self.lr = my_config['lr']
        self.warmup_steps = my_config['warmup_steps']

        self.device = my_config['device']
        self.ae.to(self.device)

        # Resampling setup
        self.resample_steps = my_config['resample_steps']
        self.resample_training_steps = my_config['resample_training_steps']
        if self.resample_steps is not None:
            # Track steps since each neuron was last active
            self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None

        # Initialize optimizer with constrained decoder weights
        self.optimizer = ConstrainedAdam(
            params=self.ae.parameters(),
            constrained_params=self.ae.decoder.parameters(),
            lr=self.lr
        )
        
        # Setup learning rate warmup
        if self.resample_steps is None:
            def warmup_fn(step):
                return min(step / self.warmup_steps, 1.0)
        else:
            def warmup_fn(step):
                return min((step % self.resample_steps) / self.warmup_steps, 1.0)
            
        self.scheduler = t.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_fn
        )

        # L1 penalty annealing setup
        self.final_l1_penalty = my_config['l1_penalty']
        self.l1_annealing_steps = int(my_config['l1_annealing_steps'])
        self.current_l1_penalty = 0 if self.l1_annealing_steps > 0 else self.l1_penalty

    def update_l1_penalty(self, step):
        """Update L1 penalty according to annealing schedule."""
        if step < self.l1_annealing_steps:
            self.current_l1_penalty = self.final_l1_penalty * \
                (step / self.l1_annealing_steps)
        else:
            self.current_l1_penalty = self.final_l1_penalty

    def resample_neurons(self, deads, activations, step):
        """
        Resample inactive neurons using high-loss activations.

        Args:
            deads: Boolean tensor indicating inactive neurons
            activations: Current batch of activations for resampling
        """
        with t.no_grad():
            if deads.sum() == 0:
                return

            # Compute reconstruction loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample input vectors based on loss
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(
                losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Get average norm of active neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # Update dead neuron parameters
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:, deads] = (
                sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)
            ).T
            self.ae.encoder.bias[deads] = 0.0

            # Reset optimizer state for resampled neurons
            state_dict = self.optimizer.state_dict()["state"]
            state_dict[1]["exp_avg"][deads] = 0.0
            state_dict[1]["exp_avg_sq"][deads] = 0.0
            state_dict[2]["exp_avg"][deads] = 0.0
            state_dict[2]["exp_avg_sq"][deads] = 0.0
            state_dict[3]["exp_avg"][:, deads] = 0.0
            state_dict[3]["exp_avg_sq"][:, deads] = 0.0

    @property
    def current_lr(self):
        """Get current optimizer learning rate."""
        return self.optimizer.param_groups[0]["lr"]


    def loss(self, x, logging=False, **kwargs):
        """
        Compute loss for current batch.

        Args:
            x: Input activations
            logging: Whether to return extended logging information

        Returns:
            If logging=False: Combined loss value
            If logging=True: Named tuple with reconstruction details and losses
        """
        
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        # local similarity loss
        nearby_difference = (t.cat([f[1:-2].unsqueeze(0), f[2:-1].unsqueeze(0), f[3:].unsqueeze(0)], dim=0) - f[0:-3].unsqueeze(0)).norm(p=1, dim=-1)
        local_similarity_loss = (self.softmin(nearby_difference)*nearby_difference).sum(dim=0).mean()

        if self.smooth_penalty > 0:
            loss = l2_loss + self.current_l1_penalty * (l1_loss + local_similarity_loss*self.smooth_penalty)
        else:
            loss = l2_loss + self.current_l1_penalty * l1_loss

        loss_log = {
                "l2_loss": l2_loss.item() / 1280, # 1280 is the dimension of the ESM2-650M embedding
                "mse_loss": (x - x_hat).pow(2).mean().item(),
                "sparsity_loss": l1_loss.item(),
                "local_similarity_loss": local_similarity_loss.item(),
                "loss": loss.item(),
                "lr": self.current_lr,
                "l1_penalty": self.current_l1_penalty}
        

        if self.steps_since_active is not None:
            # Track inactive neurons
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
            loss_log['n_dead_neuron'] = (self.steps_since_active > self.resample_steps).sum().item()


        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(x, x_hat, f, loss_log)

    def update(self, step, activations):
        """
        Perform single training step.

        Args:
            step: Current training step
            activations: Batch of input activations
        """
        if self.l1_annealing_steps > 0:
            self.update_l1_penalty(step)

        # Compute and apply gradients
        self.optimizer.zero_grad()
        
        activations = activations.to(self.device)
        loss = self.loss(activations)

        
        loss.backward()
        
        # Clip gradients by value
        t.nn.utils.clip_grad_value_(self.ae.parameters(), clip_value=0.5)

        self.optimizer.step()
        self.scheduler.step()

        # Check for dead neurons
        if self.resample_steps is not None and step % self.resample_steps == 0:
            if step <= self.resample_training_steps:
                self.resample_neurons(
                    self.steps_since_active > self.resample_steps, activations, step,
                )