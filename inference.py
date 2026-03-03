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
    
    print("\n--- Generating text ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=model.device)
    generated_indices = model.generate(context, max_new_tokens=100)[0].tolist()
    
    decode = lambda l: ''.join([datamodule.itos[i] for i in l])
    print(decode(generated_indices))


if __name__ == "__main__":
    main()
