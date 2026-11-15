# 📊 BÁO CÁO KẾT QUẢ THỰC NGHIỆM: PHASE 1 (GIẢ THUYẾT H1)
**Đề tài CITA 2027: Asynchronous Cross-Horizon Error Transfer**
*Trạng thái: Chạy độc lập 100% - Đối kháng trực tiếp với B0_Frozen*

---

### 1. Bảng số liệu chi tiết Phase 1 (S&P 500 vs BTCUSDT)
*Đo lường CRPS, MAE và Tỷ lệ bao phủ (Coverage 80%) qua 4 chân trời dự báo:*

| Tập dữ liệu | Chân trời (h) | B0_Frozen CRPS | H1 CRPS | H1 MAE | Coverage (80%) | Cải thiện CRPS (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S&P 500 (SPY)** | $h=1$ | 0.007360 | 0.006894 | 0.008779 | 50.0% | **+6.34%** |
| **S&P 500 (SPY)** | $h=4$ | 0.007051 | 0.006630 | 0.008344 | 38.8% | **+5.96%** |
| **S&P 500 (SPY)** | $h=8$ | 0.007127 | 0.006605 | 0.008209 | 35.0% | **+7.31% (Đỉnh cao)** |
| **S&P 500 (SPY)** | $h=16$ | 0.006887 | 0.006951 | 0.008409 | 38.8% | **-0.93% (Suy thoái)** |
| **SPY Trung bình** | **All** | **0.007106** | **0.006770** | **0.008435** | **40.6%** | **+4.67%** |
| **BTCUSDT (Crypto)** | $h=1$ | 0.019506 | 0.020231 | 0.023585 | 23.8% | **-3.72%** |
| **BTCUSDT (Crypto)** | $h=4$ | 0.020044 | 0.021271 | 0.024308 | 33.8% | **-6.12%** |
| **BTCUSDT (Crypto)** | $h=8$ | 0.021181 | 0.023819 | 0.027004 | 22.5% | **-12.46%** |
| **BTCUSDT (Crypto)** | $h=16$ | 0.019269 | 0.021680 | 0.024707 | 21.2% | **-12.51%** |
| **BTC Trung bình** | **All** | **0.020000** | **0.021750** | **0.024901** | **25.3%** | **-8.70% (Negative Transfer)** |

---

### 2. Biểu đồ trực quan Phase 1
Biểu đồ phân tích độ nhạy và hiệu quả prequential của Phase 1:  
[figure_h1_detailed.png](file:///D:/01_AI_Research_Projects/cita2027-phase2/paper_figures/figure_h1_detailed.png)

---

### 3. Đánh giá học thuật rút ra từ Phase 1
1. **Điểm sáng:** Trên thị trường tài sản ổn định (SPY), H1 chứng minh xuất sắc khả năng chuyển giao sai số giữa các chân trời, giúp giảm CRPS trung bình **+4.67%**, đạt đỉnh **+7.31%** tại $h=8$.
2. **Điểm nghẽn lý thuyết:** 
   - Khi độ trễ quá xa ($h=16$), việc chuyển giao sai số bị chậm nhịp khiến hiệu quả âm nhẹ (-0.93%).
   - Trên tài sản biến động cao phi tuyến (BTCUSDT), việc chuyển giao trực tiếp không kiểm soát gây ra hiện tượng **Negative Transfer nghiêm trọng (-8.70%)**.
   - $\rightarrow$ **Kết luận:** Bắt buộc phải có cơ chế phân rã độ tuổi (Age Decay) ở Phase 2 để lọc nhiễu phản hồi trễ.
