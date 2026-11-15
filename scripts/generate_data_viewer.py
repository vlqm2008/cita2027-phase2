import pandas as pd
import json

df = pd.read_csv(r"D:\01_AI_Research_Projects\cita2027-phase2\artifacts\processed\SPY_20261104T095117433396Z_b41a8489.csv")

cols = ['date', 'close', 'adj_close', 'log_return', 'split']
data = df[cols].fillna('').values.tolist()

html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>S&P 500 (SPY) Dataset Viewer - SpectralTFT Base Paper</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 24px; background-color: #0f172a; color: #f8fafc; }
        h1 { color: #38bdf8; margin-bottom: 6px; font-size: 24px; }
        .meta { color: #94a3b8; font-size: 14px; margin-bottom: 24px; }
        .badge { padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 12px; display: inline-block; }
        .split-warmup { background: #334155; color: #cbd5e1; }
        .split-development { background: #7f1d1d; color: #fecaca; }
        .split-validation { background: #78350f; color: #fef3c7; }
        .split-final_test { background: #14532d; color: #bbf7d0; }
        .dataTables_wrapper { background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; }
        table.dataTable { color: #f8fafc !important; }
        table.dataTable thead th { color: #38bdf8 !important; border-bottom: 1px solid #334155 !important; }
        table.dataTable tbody tr { background-color: #1e293b !important; }
        table.dataTable tbody tr:hover { background-color: #334155 !important; }
        .dataTables_info, .dataTables_length, .dataTables_filter { color: #94a3b8 !important; }
        input[type="search"], select { background: #0f172a; border: 1px solid #475569; color: #f8fafc; padding: 6px 10px; border-radius: 6px; }
    </style>
</head>
<body>
    <h1>📊 Dữ liệu S&P 500 (SPY) - Bài báo gốc SpectralTFT & SpectralTFT-U</h1>
    <div class="meta">
        Tổng số: <strong>__LEN__</strong> ngày giao dịch (2005 - 2026) | 
        Biến mục tiêu dự báo: <code>log_return</code> | 
        Mô hình áp dụng: <strong>Temporal Fusion Transformer + Phạt phổ Fourier</strong>
    </div>

    <table id="dataTable" class="display" style="width:100%">
        <thead>
            <tr>
                <th>Ngày (Date)</th>
                <th>Giá đóng cửa (Close USD)</th>
                <th>Giá điều chỉnh (Adj Close)</th>
                <th>Lợi suất log (Log Return)</th>
                <th>Tập phân chia (Split)</th>
            </tr>
        </thead>
    </table>

    <script>
        const tableData = __DATA__;
        $(document).ready(function() {
            $('#dataTable').DataTable({
                data: tableData,
                pageLength: 25,
                order: [[0, 'desc']],
                columnDefs: [
                    {
                        targets: 1,
                        render: function(data) {
                            return '$' + parseFloat(data).toFixed(2);
                        }
                    },
                    {
                        targets: 2,
                        render: function(data) {
                            return '$' + parseFloat(data).toFixed(2);
                        }
                    },
                    {
                        targets: 3,
                        render: function(data) {
                            if (data === '') return '-';
                            let val = parseFloat(data);
                            let color = val >= 0 ? '#4ade80' : '#f87171';
                            return '<span style="color:' + color + '; font-weight:600;">' + (val > 0 ? '+' : '') + val.toFixed(6) + '</span>';
                        }
                    },
                    {
                        targets: 4,
                        render: function(data) {
                            return '<span class="badge split-' + data + '">' + data + '</span>';
                        }
                    }
                ]
            });
        });
    </script>
</body>
</html>
""".replace("__LEN__", f"{len(df):,}").replace("__DATA__", json.dumps(data))

with open(r"D:\01_AI_Research_Projects\cita2027-phase2\view_data.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Generated D:\\01_AI_Research_Projects\\cita2027-phase2\\view_data.html successfully")
