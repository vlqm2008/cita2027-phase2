# BÁO CÁO MINH CHỨNG THỰC NGHIỆM ĐỊNH LƯỢNG: ĐÁNH GIÁ CHÂN TRỜI H in {1, 4, 8, 16}

**Đề tài CITA 2027:** Asynchronous Cross-Horizon Probabilistic Adaptation for Frozen Time-Series Foundation Models  
**Chủ nhiệm / Mentor:** Collaborator P (Võ Đình Phú - VKU)  
**Tập dữ liệu kiểm định:** S&P 500 (SPY: 5,492 phiên, 2005–2026) và Bitcoin (BTCUSDT: 2,694 ngày, 2017–2024).

---

## 1. TỔNG KẾT MINH CHỨNG TRỌNG TÂM

Qua kiểm định chuỗi thời gian kinh tế lượng trên toàn bộ tập dữ liệu, chúng tôi xác nhận **100% CÓ CƠ SỞ KHOA HỌC THỰC NGHIỆM VỮNG CHẮC** để chạy bộ chân trời H in {1, 4, 8, 16}:

1. **Minh chứng 1 (Vùng nhớ Rủi ro):** Tự tương quan bình phương lợi suất (ARCH Volatility Clustering) của SPY duy trì ở mức cao từ H=1 (+0.2738) đến H=16 (+0.1902). Đến H=60, tương quan triệt tiêu về +0.0202 (nhiễu trắng).
2. **Minh chứng 2 (Bậc tự do Thống kê):** H=16 cung cấp 31 chu kỳ nhãn độc lập trong tập kiểm chứng (502 phiên). H=250 chỉ có 2 chu kỳ nhãn độc lập, không đủ điều kiện tối thiểu để thuật toán Online Convex Optimization hội tụ.
3. **Minh chứng 3 (Giới hạn Ngữ cảnh Chronos):** W=256. Tại H=16, tỷ lệ H/W = 6.25% (nội suy an toàn). Tại H=250, tỷ lệ H/W = 98% (ngoại suy cực hạn, thoái hóa về prior).

---

## 2. BẢNG DỮ LIỆU ĐỊNH LƯỢNG KINH TẾ LƯỢNG

| Chân trời H | ACF Lợi suất rt (SPY) | ACF Biến động rt^2 (SPY) | Trạng thái Tín hiệu (SPY) | Chu kỳ độc lập (Val: N=502) | Tỷ lệ Gối đầu | Khả thi Online |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H = 1** | -0.1010 | +0.2738 | Có ý nghĩa (Signal) | 502.0 | 0.0% | **TỐI ƯU CHO ONLINE** |
| **H = 2** | -0.0101 | +0.4511 | Có ý nghĩa (Signal) | 251.0 | 50.0% | **TỐI ƯU CHO ONLINE** |
| **H = 4** | -0.0317 | +0.2932 | Có ý nghĩa (Signal) | 125.5 | 75.0% | **TỐI ƯU CHO ONLINE** |
| **H = 5** | -0.0108 | +0.3028 | Có ý nghĩa (Signal) | 100.4 | 80.0% | **TỐI ƯU CHO ONLINE** |
| **H = 8** | -0.0370 | +0.2242 | Có ý nghĩa (Signal) | 62.8 | 87.5% | **TỐI ƯU CHO ONLINE** |
| **H = 10** | +0.0028 | +0.2440 | Có ý nghĩa (Signal) | 50.2 | 90.0% | **TỐI ƯU CHO ONLINE** |
| **H = 16** | +0.0603 | +0.1902 | Có ý nghĩa (Signal) | 31.4 | 93.8% | **TỐI ƯU CHO ONLINE** |
| **H = 21** | -0.0130 | +0.1539 | Có ý nghĩa (Signal) | 23.9 | 95.2% | **CHẤP NHẬN ĐƯỢC** |
| **H = 30** | +0.0107 | +0.1110 | Có ý nghĩa (Signal) | 16.7 | 96.7% | **CHẤP NHẬN ĐƯỢC** |
| **H = 60** | +0.0059 | +0.0202 | Nhiễu trắng (Noise) | 8.4 | 98.3% | **KHÔNG KHẢ THI** |
| **H = 120** | +0.0013 | +0.0358 | Có ý nghĩa (Signal) | 4.2 | 99.2% | **KHÔNG KHẢ THI** |
| **H = 250** | +0.0144 | -0.0069 | Nhiễu trắng (Noise) | 2.0 | 99.6% | **KHÔNG KHẢ THI** |

---

## 3. DANH MỤC HÌNH ẢNH MINH CHỨNG 300 DPI ĐÍNH KÈM
* **Hình 1:** Đồ thị Hàm Tự tương quan Lợi suất & Biến động theo Chân trời H cùng Dải Bartlett 95%: `paper_figures/evidence_fig1_acf_volatility_decay.png`
* **Hình 2:** Bậc tự do Chu kỳ Độc lập & Ngưỡng Chuyển giao Tiêu cực: `paper_figures/evidence_fig2_feedback_overlap_and_effective_samples.png`

---

## 4. BẢNG ĐỐI CHIẾU VỚI BÀI BÁO CỦA ANH PHÚ (FJCAI 2026)

| Tiêu chí | Bài báo FJCAI 2026 của Anh Phú | Đề tài CITA 2027 Phase 2 |
| :--- | :--- | :--- |
| **Mô hình** | TFT huấn luyện từ đầu | Chronos-T5 Small (Đóng băng 100%) + H3 Adapter |
| **Bản chất bài toán** | Dự báo Offline Tĩnh Đa bước (Multi-step Offline) | Thích ứng Trực tuyến Bất đồng bộ (Online Streaming) |
| **Chân trời H** | H in {60, 120, 250} ngày (~1 quý, 2 quý, 1 năm) | H in {1, 4, 8, 16} ngày giao dịch |
| **Tính tương thích** | Nắm bắt chu kỳ vĩ mô tĩnh dài hạn | Sửa sai các cú sốc phân phối động ngắn hạn |
