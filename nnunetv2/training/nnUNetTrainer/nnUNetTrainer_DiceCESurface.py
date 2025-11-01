import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dc_ce_surface_loss import DC_CE_Surface_Combined


class nnUNetTrainer_SurfaceRefineEarlyStop(nnUNetTrainer):
    """
    Phase-2 refinement trainer:
      - starts from pretrained_weights passed via --pretrained_weights
      - lowers LR at start (fine-tune)
      - uses Dice+CE+Surface loss (boundary-aware)
      - stops early if val doesn't improve for N epochs
    """

    # tuning knobs:
    patience_epochs = 20        # early stop patience
    lr_scale_on_start = 0.1     # multiply LR by this at fine-tune start
    w_dcce = 1.0                # weight for Dice+CE block
    w_surf = 0.5                # weight for surface loss
    ignore_background_for_surface = False

    def build_loss(self):
        """
        nnU-Net calls this during initialization.
        We build Dice+CE+Surface with nnU-Net's own DC_and_CE_loss.
        """
        sd_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': True,     # include background in Dice. You may flip if you want.
            'ddp': self.is_ddp
        }
        ce_kwargs = {
            'weight': None,    # class weights; could add tumor class boost here if very imbalanced
        }

        self.loss = DC_CE_Surface_Combined(
            soft_dice_kwargs=sd_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate_function=torch.mean,
            w_dcce=self.w_dcce,
            w_surf=self.w_surf,
            ignore_background_for_surface=self.ignore_background_for_surface
        )

    def on_train_start(self):
        """
        Called once before epochs loop.
        We'll:
        - call parent (sets optimizer etc.)
        - downscale LR for fine-tuning
        - init patience counters
        """
        super().on_train_start()

        # LR downscale for fine-tune stability
        for g in self.optimizer.param_groups:
            g['lr'] *= self.lr_scale_on_start

        # Init early stopping bookkeeping
        self._best_val_metric_so_far = -1e9
        self._epochs_since_improve = 0

    def run_training(self):
        """
        Custom training loop with patience-based early stopping.
        Mirrors nnUNetTrainer.run_training() but injects stopping condition.
        """
        self.on_train_start()

        while self.epoch < self.num_epochs:
            # ---- training epoch ----
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))

            # ---- validation ----
            # This writes val preds, updates best model tracking, etc.
            self.perform_actual_validation(export_validation_probabilities=False)

            # nnU-Net tracks validation score internally.
            # Depending on your nnU-Net version, the attribute names are:
            #   self.best_val_eval_criterion_MA (moving average of val metric)
            #   self.best_val_eval_criterion     (raw val metric)
            # We try MA first, fallback to non-MA.
            if hasattr(self, "best_val_eval_criterion_MA") and self.best_val_eval_criterion_MA is not None:
                current_val = self.best_val_eval_criterion_MA
            elif hasattr(self, "best_val_eval_criterion") and self.best_val_eval_criterion is not None:
                current_val = self.best_val_eval_criterion
            else:
                current_val = None

            # ---- patience update ----
            if current_val is not None:
                if current_val > self._best_val_metric_so_far:
                    self._best_val_metric_so_far = current_val
                    self._epochs_since_improve = 0
                else:
                    self._epochs_since_improve += 1

            # next epoch
            self.epoch += 1

            # ---- early stop condition ----
            if self._epochs_since_improve > self.patience_epochs:
                print(f"[EarlyStop] No val improvement for {self.patience_epochs} epochs. Stopping.")
                break

        # wrap up: save final etc.
        self.on_train_end()
