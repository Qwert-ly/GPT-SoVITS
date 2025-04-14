# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import os
import logging
from pathlib import Path
import platform
from collections import OrderedDict
from typing import Dict, Any, Optional

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from AR.utils import get_newest_ckpt
from process_ckpt import my_save

# Set environment variables
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
            self,
            config: Dict[str, Any],
            if_save_latest: bool,
            if_save_every_weights: bool,
            half_weights_save_dir: str,
            exp_name: str,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if not self._should_save_on_train_epoch_end(trainer):
            return

        monitor_candidates = self._monitor_candidates(trainer)

        # Check if we should save a checkpoint based on epoch count
        if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            # Clean previous checkpoints if needed
            if self.if_save_latest:
                to_clean = list(os.listdir(self.dirpath))

            # Save new checkpoint
            self._save_topk_checkpoint(trainer, monitor_candidates)

            # Remove old checkpoints if only keeping latest
            if self.if_save_latest:
                for name in to_clean:
                    try:
                        os.remove(f"{self.dirpath}/{name}")
                    except:
                        pass

            if self.if_save_every_weights and os.environ.get("LOCAL_RANK", "0") == "0":
                self._save_half_precision_weights(trainer)

        self._save_last_checkpoint(trainer, monitor_candidates)

    def _save_half_precision_weights(self, trainer):
        """Save model weights in half precision."""
        to_save = {
            "weight": OrderedDict((k, v.half()) for k, v in trainer.strategy.lightning_module.state_dict().items()),
            "config": self.config,
            "info": f"GPT-e{trainer.current_epoch + 1}"
        }

        checkpoint_path = f"{self.half_weights_save_dir}/{self.exp_name}-e{trainer.current_epoch + 1}.ckpt"
        my_save(to_save, checkpoint_path)


def setup_environment():
    """Set up environment variables for distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"


def create_trainer(config: Dict[str, Any], ckpt_callback: ModelCheckpoint, logger: TensorBoardLogger) -> Trainer:
    """Create and configure a PyTorch Lightning Trainer."""
    using_gpu = torch.cuda.is_available()
    is_windows = platform.system() == "Windows"

    return Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if using_gpu else "cpu",
        limit_val_batches=0,  # Disable validation
        devices=-1 if using_gpu else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(process_group_backend="nccl" if not is_windows else "gloo") if using_gpu else "auto",
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,  # Fixes training step inconsistency with custom bucket_sampler
    )


def setup_directories(config: Dict[str, Any]) -> tuple[Path, Path]:
    """Create necessary output directories."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, ckpt_dir


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint in the checkpoint directory."""
    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        return ckpt_dir / newest_ckpt_name
    except Exception:
        return None


def main(args):
    config = load_yaml_config(args.config_file)

    output_dir, ckpt_dir = setup_directories(config)

    # Set random seed
    seed_everything(config["train"]["seed"], workers=True)

    # Create checkpoint callback
    ckpt_callback = CustomModelCheckpoint(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    setup_environment()

    # Create trainer
    trainer = create_trainer(config, ckpt_callback, logger)

    # Create model and data module
    model = Text2SemanticLightningModule(config, output_dir)
    data_module = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
    )

    # Find latest checkpoint
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    print(f"ckpt_path: {ckpt_path}")

    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="配置文件路径",
    )

    args = parser.parse_args()
    logging.info(str(args))
    main(args)
