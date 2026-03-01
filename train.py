# train.py
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import CharDataModule
from src.model import GPTLightningModule


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Khởi tạo seed
    pl.seed_everything(cfg.seed)

    # Khởi tạo DataModule
    datamodule = CharDataModule(cfg)
    
    # Chúng ta phải gọi setup() thủ công ở đây để lấy vocab_size cho model
    datamodule.setup()
    vocab_size = datamodule.vocab_size

    # Khởi tạo Model
    model = GPTLightningModule(vocab_size=vocab_size, cfg=cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',         # Theo dõi giá trị val_loss
        dirpath='checkpoints/',    # Thư mục lưu
        filename='gpt-shakespeare-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,               # Chỉ lưu 1 bản tốt nhất
        mode='min',                 # Lưu khi val_loss thấp nhất
    )

    # Khởi tạo Trainer
    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        callbacks=[checkpoint_callback]
    )

    # Bắt đầu training
    trainer.fit(model, datamodule=datamodule)

    if trainer.global_rank == 0:
        # Cách 2: Chỉ lưu trọng số của mô hình PyTorch thuần (phổ biến hơn để load lại sau này)
        torch.save(model.state_dict(), "model_weights.pth")
        print("Model weights saved successfully!")

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