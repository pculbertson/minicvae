import torch
from torch.utils.data import DataLoader
from safety_filter.learning.models import CVAE
from safety_filter.learning.DI.data import DIDataset
from safety_filter.learning.cvae_utils import elbo_loss
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import yaml


@dataclass
class TrainConfig:
    # Model architecture
    output_dim: int
    latent_dim: int
    cond_dim: int
    encoder_layers: list
    decoder_layers: list
    prior_layers: list

    # Training hyperparams
    batches_per_epoch: int
    epochs: int
    step_size: int
    gamma: float
    lr: float
    save_epochs: int
    val_epochs: int
    device: str


# Specify the training configuration.
TRAIN_CONFIG = TrainConfig(
    output_dim=2,
    latent_dim=2,
    cond_dim=2,
    encoder_layers=[32, 32, 32, 32],
    decoder_layers=[32, 32, 32, 32],
    prior_layers=[32, 32],
    batches_per_epoch=20,
    epochs=300,
    step_size=50,  # steps per decay for lr scheduler
    gamma=0.75,  # multiplicative decay for lr scheduler
    lr=1e-3,
    save_epochs=100,
    val_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


# Specify the path to the dataset.
TRAIN_DATASET_PATH = "/home/preston/ADAM-ROS/src/safety_filter/src/safety_filter/learning/DI/data/DI_20240319-153001.pt"

VAL_DATASET_PATH = "/home/preston/ADAM-ROS/src/safety_filter/src/safety_filter/learning/DI/data/DI_20240319-153026.pt"

OUTPUT_PATH = Path(
    "/home/preston/ADAM-ROS/src/safety_filter/src/safety_filter/learning/DI/models"
)


def train_loop(
    model: CVAE,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    output_dir: Path,
    config: TrainConfig,
):
    """
    Trains the CVAE model.

    Args:
        model: CVAE; the CVAE model.
        train_dataloader: DataLoader; the training dataloader.
        epochs: int; the number of epochs to train for.
    """
    # Set the model to training mode.
    model.train()

    # Create the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer=optimizer, step_size=config.step_size, gamma=config.gamma
    # )

    # Train the model.
    for epoch in range(config.epochs):
        # Train the model for one epoch.
        for batch in train_dataloader:
            # Zero the gradients.
            optimizer.zero_grad()

            # Compute the loss.
            loss = elbo_loss(
                model, batch["d"].to(config.device), batch["cond"].to(config.device)
            )

            # Backpropagate the gradients.
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update the weights.
            optimizer.step()
            # scheduler.step()

        # Print the loss.
        if epoch % config.val_epochs == 0:
            val_loss = 0
            for batch in val_dataloader:
                val_loss += elbo_loss(
                    model, batch["d"].to(config.device), batch["cond"].to(config.device)
                ).item()
            val_loss /= len(val_dataloader)
            print(f"Epoch {epoch}: train_loss={loss}, val_loss={val_loss}")

        # Save the model.
        if epoch % config.save_epochs == 0:
            torch.save(model.state_dict(), output_dir / f"model_{epoch}.pth")


if __name__ == "__main__":
    train_dataset = DIDataset(TRAIN_DATASET_PATH)
    val_dataset = DIDataset(VAL_DATASET_PATH)

    batch_size = int(len(train_dataset) / TRAIN_CONFIG.batches_per_epoch)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create the output directory.
    output_dir = OUTPUT_PATH / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dump the training configuration.
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(TRAIN_CONFIG), f)

    model = CVAE(
        output_dim=TRAIN_CONFIG.output_dim,
        latent_dim=TRAIN_CONFIG.latent_dim,
        cond_dim=TRAIN_CONFIG.cond_dim,
        encoder_layers=TRAIN_CONFIG.encoder_layers,
        decoder_layers=TRAIN_CONFIG.decoder_layers,
        prior_layers=TRAIN_CONFIG.prior_layers,
        cond_mean=train_dataset.cond_mean,
        cond_var=train_dataset.cond_var,
        output_mean=train_dataset.d_mean,
        output_var=train_dataset.d_var,
    ).to(TRAIN_CONFIG.device)

    train_loop(model, train_dataloader, val_dataloader, output_dir, TRAIN_CONFIG)
    torch.save(model.state_dict(), output_dir / f"model_{TRAIN_CONFIG.epochs}.pth")