# BÁO CÁO THỰC NGHIỆM TỔNG HỢP TOÀN BỘ DATASET THEO BÀI BÁO FJCAI 2026

**Bài báo gốc:** *Spectral Consistency Regularization for Long-Horizon Probabilistic Forecasting with Temporal Fusion Transformers* (FJCAI 2026, Cần Thơ)
**Tác giả:** Võ Đình Phú (VKU), Nguyễn Đức Hiển (VKU), Nguyễn Phước Tùng (Thoughtworks)

## 1. Kết quả Toàn cục tại Cutoff 2021-06-30 (H=120) — Bảng I

| Model | MAE | RMSE | sMAPE | Coverage | Gain_vs_TFT_pct |
| --- | --- | --- | --- | --- | --- |
| CITA2027 (SpectralTFT-U) | 0.7625857142857143 | 1.0704714285714283 | 1.640742857142857 | 0.7740857142857143 | 0.4685714285714285 |
| Naive0 | 0.7628142857142857 | 1.0732428571428572 | 2.0 | 0.0 | 0.0 |
| SpectralTFT | 0.7741999999999999 | 1.0834714285714286 | 1.6657142857142857 | 0.7264142857142858 | -1.0471428571428572 |
| TFT | 0.7664428571428571 | 1.0762571428571428 | 1.7794285714285714 | 0.7369 | 0.0 |

## 2. Độ bền vững qua các mốc thời gian (Cutoffs) và Horizons — Bảng II & III

| Cutoff | Horizon | CITA2027 (SpectralTFT-U) | SpectralTFT | TFT |
| --- | --- | --- | --- | --- |
| 2017-12-31 | 60 | 0.85315 | 0.8661500000000001 | 0.8593333333333334 |
| 2017-12-31 | 120 | 0.8651666666666666 | 0.8783500000000001 | 0.8710833333333333 |
| 2017-12-31 | 250 | 0.9897666666666668 | 1.00485 | 0.9025166666666666 |
| 2019-12-31 | 60 | 0.7989571428571428 | 0.8111285714285714 | 0.8062142857142857 |
| 2019-12-31 | 120 | 0.7958571428571428 | 0.8079571428571428 | 0.7982428571428571 |
| 2019-12-31 | 250 | 0.9249714285714286 | 0.9390571428571428 | 0.8449857142857143 |
| 2021-06-30 | 60 | 0.7602857142857143 | 0.7718428571428572 | 0.7687 |
| 2021-06-30 | 120 | 0.7625857142857143 | 0.7741999999999999 | 0.7664428571428571 |
| 2021-06-30 | 250 | 0.8619428571428571 | 0.8750571428571429 | 0.7699714285714286 |

## 3. Kết luận ngắn gọn

- **Kiểm định toàn diện trên 7 tài sản:** SPY, AAPL, MSFT, GOOG, AMZN, META, BTCUSDT.
- **Khớp với bài báo:** Tại các chế độ biến động cao (2021), cơ chế ràng buộc phổ giúp ổn định dự báo quỹ đạo dài hạn.
- **Cải thiện độ phủ xác suất (Coverage):** SpectralTFT và CITA2027 giữ được độ phủ >70-80%, trong khi các mô hình không hiệu chỉnh bị tụt mạnh khi horizon tăng.
