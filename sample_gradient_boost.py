import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize as opt

# =====================================================================
# 1. CẤU HÌNH HYPERPARAMETERS & THIẾT LẬP MÔI TRƯỜNG
# =====================================================================
# Giả định kích thước cho bài toán Text Generation
B = 2        # Batch size (Số lượng chuỗi văn bản)
L = 16       # Sequence Length (Chiều dài chuỗi token)
V = 50000    # Vocab Size (Kích thước từ vựng)
D = 768      # Hidden dimension (Kích thước nhúng)

num_models = 3     # Số lượng Weak Learners (N)
epochs = 10        # Số vòng huấn luyện cho mỗi Weak Learner
shrinkage_nu = 1.0 # Nhả phanh hoàn toàn (100%) vì ta chỉ dùng 3 models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# 2. HÀM TÍNH ĐẠO HÀM CROSS-ENTROPY (TẠO VECTOR LA BÀN W)
# =====================================================================
def compute_ce_residual(ensemble_logits, labels, vocab_size):
    """
    Tính đạo hàm âm của Cross-Entropy Loss theo Logits.
    Hàm này không dùng nn.CrossEntropyLoss, mà tính toán thẳng giải tích: w = Y - P
    
    Inputs:
        ensemble_logits: Tensor shape - Dự đoán tổng hợp của quần thể hiện tại
        labels: Tensor shape - Ground truth (các token đích bị dịch phải 1 nhịp)
    Outputs:
        w: Tensor shape - Vector thặng dư (Target cho hàm MSE của mô hình g)
    """
    # Bước a: Biến tập Logits thành Xác suất P (Softmax)
    probs = F.softmax(ensemble_logits, dim=-1) # Shape:
    
    # Bước b: Biến nhãn đúng thành One-hot vector Y
    Y_onehot = F.one_hot(labels, num_classes=vocab_size).float() # Shape:
    
    # Bước c: Đạo hàm âm (Negative Gradient) chính là phép trừ Y - P
    w = Y_onehot - probs
    return w

# =====================================================================
# 3. HÀM MỤC TIÊU CHO THUẬT TOÁN LINE SEARCH
# =====================================================================
def ce_line_search_objective(alpha, current_f, current_g, labels):
    """
    Hàm này được dùng để thuật toán Scipy quét tìm ra alpha làm cho
    Cross-Entropy của toàn bộ hệ thống chạm đáy.
    """
    # Thử nghiệm một bước nhảy: f_new = f_old + alpha * g(x)
    f_new = current_f + alpha * current_g # Shape:
    
    # Để dùng nn.CrossEntropyLoss của PyTorch, ta phải làm phẳng Tensor
    f_new_flat = f_new.view(-1, f_new.size(-1)) # Shape:
    labels_flat = labels.view(-1)               # Shape:
    
    # Tính Loss
    loss = F.cross_entropy(f_new_flat, labels_flat)
    return loss.item()

# =====================================================================
# 4. MÔ PHỎNG WEAK LEARNER (MẠNG GPT-2 2-LAYERS THU GỌN)
# =====================================================================
class DummyGPT2WeakLearner(nn.Module):
    def __init__(self):
        super().__init__()
        # Để code chạy được nhanh, ta mô phỏng mạng Transformer bằng một khối Linear.
        # Trong thực tế, đây sẽ là cấu trúc của GPT-2 với Self-Attention.
        self.net = nn.Sequential(
            nn.Linear(D, 2048),
            nn.GELU(),
            nn.Linear(2048, V)
        )
        
    def forward(self, x):
        # Input shape: -> Output shape:
        return self.net(x)

# =====================================================================
# 5. KHỞI TẠO QUẦN THỂ VÀ DỮ LIỆU
# =====================================================================
# Dữ liệu giả lập
inputs = torch.randn(B, L, D).to(device)
labels = torch.randint(0, V, (B, L)).to(device)

# BƯỚC QUAN TRỌNG: Khởi tạo f_0(x) = 0 cho mọi token và mọi class
ensemble_logits = torch.zeros(B, L, V).to(device)

# Tạo danh sách các Weak Learners và hàm Loss MSE
weak_learners =
mse_loss_fn = nn.MSELoss()

# =====================================================================
# 6. VÒNG LẶP HUẤN LUYỆN BOOSTING (TRAINING LOOP)
# =====================================================================
print("Bắt đầu quy trình huấn luyện BoostGPT...")

for t in range(num_models):
    print(f"\\n=== Training Weak Learner {t+1}/{num_models} ===")
    model = weak_learners[t]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # BƯỚC 6.1: Tính thặng dư La bàn w(x) 
    # (Lưu ý: Chỉ tính 1 lần duy nhất ở đầu mỗi vòng lặp Boosting)
    with torch.no_grad():
        target_w = compute_ce_residual(ensemble_logits, labels, V)
        
    # BƯỚC 6.2: Train Weak Learner khớp với w(x) bằng hàm MSE
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Mô hình g_t(x) đưa ra dự đoán
        g_x = model(inputs) # Shape:
        
        # Ép mô hình học vector La bàn bằng Hồi quy L2 (MSE)
        loss = mse_loss_fn(g_x, target_w)
        
        loss.backward()
        optimizer.step()
        
    print(f" -> MSE Loss của mô hình sau {epochs} epochs: {loss.item():.6f}")
    
    # BƯỚC 6.3: Dò đường Line Search tìm Alpha tối ưu
    # Lấy đầu ra cố định của g_t(x)
    detached_g_x = model(inputs).detach()
    
    # Định nghĩa lại hàm 1 biến (alpha) cho Scipy
    def line_search_fn(alpha_scalar):
        return ce_line_search_objective(alpha_scalar, ensemble_logits, detached_g_x, labels)
        
    # Chạy thuật toán giải tích số để tìm cực tiểu
    res = opt.minimize_scalar(line_search_fn)
    alpha_t = res.x
    print(f" -> Kích thước bước nhảy (Alpha) tối ưu tìm được: {alpha_t:.4f}")
    
    # BƯỚC 6.4: Cập nhật f(x) mới cho quần thể
    with torch.no_grad():
        # f_mới = f_cũ + nu * alpha * g(x)
        ensemble_logits += shrinkage_nu * alpha_t * detached_g_x
        
    # (Tùy chọn) In ra Cross-Entropy Loss hiện tại của cả hệ thống để kiểm chứng
    current_ce = ce_line_search_objective(0.0, ensemble_logits, torch.zeros_like(ensemble_logits), labels)
    print(f" -> Cross-Entropy của toàn hệ thống sau vòng {t+1} đã giảm xuống: {current_ce:.4f}")

print("\\nHoàn tất Huấn luyện BoostGPT!")