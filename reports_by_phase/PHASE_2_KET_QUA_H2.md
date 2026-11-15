# 📊 BÁO CÁO KẾT QUẢ THỰC NGHIỆM: PHASE 2 (GIẢ THUYẾT H2)
**Đề tài CITA 2027: Age-Decayed Stale Feedback Mitigation**
*Trạng thái: Chạy độc lập 100% - Đối kháng trực tiếp với B0_Frozen và H1*

---

### 1. Bảng số liệu chi tiết Phase 2 (H2 vs H1 vs B0)
*Đo lường CRPS, MAE và Tỷ lệ bao phủ qua 4 chân trời dự báo:*

| Tập dữ liệu | Chân trời (h) | B0_Frozen CRPS | H1 CRPS | H2 CRPS | H2 Coverage | Cải thiện vs B0 (%) | Hiệu quả vs H1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S&P 500 (SPY)** | $h=1$ | 0.007360 | 0.006894 | 0.006718 | 48.8% | **+8.73%** | **+2.55%** |
| **S&P 500 (SPY)** | $h=4$ | 0.007051 | 0.006630 | 0.006442 | 40.0% | **+8.63%** | **+2.84%** |
| **S&P 500 (SPY)** | $h=8$ | 0.007127 | 0.006605 | 0.006574 | 35.0% | **+7.75%** | **+0.47%** |
| **S&P 500 (SPY)** | $h=16$ | 0.006887 | 0.006951 | 0.006632 | 36.2% | **+3.70%** | **+4.59% (Cứu vãn $h=16$)** |
| **SPY Trung bình** | **All** | **0.007106** | **0.006770** | **0.006592** | **40.0%** | **+7.20%** | **+2.63% vs H1** |
| **BTCUSDT (Crypto)** | $h=1$ | 0.019506 | 0.020231 | 0.019614 | 31.2% | -0.56% | Giảm 85% tổn thất |
| **BTCUSDT (Crypto)** | $h=4$ | 0.020044 | 0.021271 | 0.020551 | 28.7% | -2.53% | Giảm 59% tổn thất |
| **BTCUSDT (Crypto)** | $h=8$ | 0.021181 | 0.023819 | 0.022048 | 22.5% | -4.10% | Giảm 67% tổn thất |
| **BTCUSDT (Crypto)** | $h=16$ | 0.019269 | 0.021680 | 0.020507 | 15.0% | -6.42% | Giảm 49% tổn thất |
| **BTC Trung bình** | **All** | **0.020000** | **0.021750** | **0.020680** | **24.4%** | **-3.40%** | **Triệt tiêu >60% thiệt hại H1** |

---

### 2. Biểu đồ trực quan Phase 2
Biểu đồ phân tích độ nhạy tham số suy giảm độ tuổi $\gamma$ của Phase 2:  
[figure_h2_detailed.png](file:///D:/01_AI_Research_Projects/cita2027-phase2/paper_figures/figure_h2_detailed.png)

---

### 3. Đánh giá học thuật rút ra từ Phase 2
1. **Giải cứu chân trời xa ($h=16$):** Nhờ hàm phân rã lũy thừa $e^{-\gamma \cdot \text{age}}$, tín hiệu trễ của chân trời $h=16$ được chiết khấu hợp lý, giúp SPY $h=16$ từ mức lỗ -0.93% đảo chiều thành **+3.70%** (tăng ròng +4.59% so với H1).
2. **Hạn chế thiệt hại thị trường biến động mạnh:** Trên BTCUSDT, mức tổn thất giảm mạnh từ -8.70% xuống chỉ còn -3.40%, chặn đứng hiện tượng sụp đổ mô hình.
3. **Mở đường cho Phase 3:** Mặc dù H2 đã giảm thiểu đáng kể lỗi trễ, sự gối đè (overlap) giữa các bước vẫn làm tăng phương sai gradient, cần chuẩn hóa Kish ở Phase 3.
