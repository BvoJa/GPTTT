import torch
import hydra
from omegaconf import DictConfig
from src.data import CharDataModule
from src.model import GPTLightningModule


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Khởi tạo DataModule
    datamodule = CharDataModule(cfg)
    
    # Chúng ta phải gọi setup() thủ công ở đây để lấy vocab_size cho model
    datamodule.setup()
    vocab_size = datamodule.vocab_size

    # Khởi tạo Model
    model = GPTLightningModule(vocab_size=vocab_size, cfg=cfg)

    # Tạo văn bản sau khi train xong
    print("\n--- Generating text ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=model.device)
    generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
    
    # Sử dụng hàm decode từ datamodule (ít dùng dict itos)
    decode = lambda l: ''.join([datamodule.itos[i] for i in l])
    print(decode(generated_indices))

if __name__ == "__main__":
    main()