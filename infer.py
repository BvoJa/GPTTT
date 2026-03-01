import torch
import hydra
from omegaconf import DictConfig
from src.data import CharDataModule
from src.model import GPTLightningModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Khởi tạo DataModule để lấy vocab_size
    datamodule = CharDataModule(cfg)
    datamodule.setup()
    vocab_size = datamodule.vocab_size

    # 2. Xác định thiết bị (Tự động chọn GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running inference on: {device} ---")

    # 3. Khởi tạo Model và nạp Trọng số
    # Cách A: Nếu bạn dùng file .pth (chỉ chứa state_dict)
    model = GPTLightningModule(vocab_size=vocab_size, cfg=cfg)
    state_dict = torch.load("model_weights.pth", map_location=device)
    model.load_state_dict(state_dict)
    
    # Cách B: Nếu bạn dùng file .ckpt (tốt hơn vì nạp được cả hyperparams)
    # model = GPTLightningModule.load_from_checkpoint("checkpoints/last.ckpt", vocab_size=vocab_size, cfg=cfg)

    # 4. Chuyển model sang chế độ eval và đúng device
    model.to(device)
    model.eval()

    # 5. Tạo văn bản (Inference)
    print("\n--- Generating text ---")
    
    # Đảm bảo context (input đầu tiên) cũng phải nằm trên cùng device với model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Sử dụng hàm generate đã định nghĩa trong GPTLightningModule
    # Thêm torch.no_grad() để giải phóng bộ nhớ và tăng tốc
    with torch.no_grad():
        generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
    
    # 6. Giải mã kết quả
    decode = lambda l: ''.join([datamodule.itos[i] for i in l])
    print(decode(generated_indices))

if __name__ == "__main__":
    main()