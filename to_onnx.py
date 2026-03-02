import torch
import hydra
import omegaconf
from omegaconf import DictConfig
from src.data import CharDataModule
from src.model import GPTLightningModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    checkpoint_path = "weights/gpt-shakespeare-epoch=00-val_loss=1.90.ckpt"

    datamodule = CharDataModule(cfg)
    datamodule.setup()

    vocab_size = datamodule.vocab_size

    model = GPTLightningModule.load_from_checkpoint(
        checkpoint_path, 
        vocab_size=vocab_size, 
        cfg=cfg, 
        weights_only = False,
    )
    model.eval()

    block_size = cfg.model.block_size
    batch_size = 1

    input_sample = torch.randint(0, vocab_size, (batch_size, block_size))

    output_path = "model.onnx"

    model.to_onnx(
        output_path,
        input_sample,
        export_params=True,
        opset_version=18,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    main()
