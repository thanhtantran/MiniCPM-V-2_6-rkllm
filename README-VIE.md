# MiniCPM-V-2_6-rkllm

Chạy mô hình ngôn ngữ hình ảnh MiniCPM-V-2.6 mạnh mẽ trên Orange Pi RK3588!

- Tốc độ suy luận (RK3588): Bộ mã hóa hình ảnh + tiền xử lý LLM + giải mã => rất nhanh
- Dung lượng bộ nhớ sử dụng (RK3588, độ dài ngữ cảnh mặc định): Bộ mã hóa hình ảnh 1.9GB + LLM 7.8GB + Hệ điều hành ~ 10.6 GB RAM (cần Orange Pi với RAM 16GB)
- Giao diện web bằng Streamlit, cho phép tải ảnh và hỏi đáp tương tác với hình ảnh.

## Hướng dẫn sử dụng

1. Clone hoặc tải repo này về máy. Mô hình không kèm theo, sẽ được tải từ Hugging Face khi chạy ứng dụng.

```bash
git clone https://github.com/thanhtantran/MiniCPM-V-2_6-rkllm
cd MiniCPM-V-2_6-rkllm
```
   
3. Phiên bản driver RKNPU2 trên bo mạch phải >=0.9.6 để chạy mô hình lớn như vậy. 
   Sử dụng lệnh sau với quyền root để kiểm tra phiên bản driver:
   
```bash
sudo cat /sys/kernel/debug/rknpu/version 
RKNPU driver: v0.9.8
```
Nếu phiên bản quá thấp, hãy tải về và cài lại bản hệ điều hành mới nhất để có driver RKNPU2 cao hơn. 
Hoặc bạn có thể bạn cần cập nhật kernel hoặc tham khảo tài liệu chính thức để biết thêm.
   
4. Cài đặt các thư viện phụ thuộc ...

```bash
pip install -r requirements.txt
```
Bạn cũng cần cài rknn-toolkit2-lite, có thể sử dụng phiên bản mới nhất bằng lệnh

```bash
pip install rknn-toolkit-lite2
```
Nếu bạn muốn cài phiên bản cụ thể của rknn-toolkit-lite2, hãy truy cập https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2 và tải về

Mô hình này sử dụng RKNN lib 1.1.4, nếu bạn chưa có hoặc có phiên bản cao hơn, hãy thay thế bằng

```bash
sudo cp lib/* /usr/lib

```

5. Chạy ứng dụng
   
```bash
streamlit run streamlit_app.py
```
Sau đó, mở trình duyệt và truy cập ứng dụng tại địa chỉ http://localhost:8501 or http://YOUR_ORANGEPI_IP:8501

Bước đầu tiên, tải mô hình bằng cách nhấn nút tương ứng. Việc này mất vài phút, bạn có thể tranh thủ uống cà phê. Chỉ cần làm một lần nếu mô hình chưa có sẵn.

![minicpm-demo1](https://github.com/user-attachments/assets/5e893143-3387-4806-87e6-f75f02313296)

Sau khi mô hình sẵn sàng, nhấn Start Inference Process và đợi đến khi giao diện trò chuyện xuất hiện.

Bỏ qua cảnh báo "Failed to start inference process. Check console for details." Không rõ vì sao cảnh báo xuất hiện, và tôi cũng lười sửa lỗi này.

6. Khi thấy giao diện 💬 Chat, nhấn Upload image để tải ảnh lên và bắt đầu tương tác.

![minicpm-demo2](https://github.com/user-attachments/assets/b8348ce2-f957-45dc-a0fd-8f1ec89efde8)

7. Đặt câu hỏi và xem phản hồi phía dưới, kèm theo bảng hiệu suất.

![minicpm-demo3](https://github.com/user-attachments/assets/c1a61f09-ca17-4893-adcd-fcd11c6b6a43)

## Tham khảo

- [sophgo/LLM-TPU models/MiniCPM-V-2_6](https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6)
- [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
- [happyme531/MiniCPM-V-2_6-rkllm](https://huggingface.co/happyme531/MiniCPM-V-2_6-rkllm)
