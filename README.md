# MiniCPM-V-2_6-rkllm

Run the Powerful MiniCPM-V-2.6 Visual Language Model on Orange Pi RK3588!

- Inference speed (RK3588): Visual encoder + LLM prefill  + decoding => very fast
- Memory usage (RK3588, default context length): Visual encoder 1.9GB + LLM 7.8GB + OS ~ 10.6 GB RAM (need to run on a 16BG RAM Orange Pi
- Streamlit web interface, upload your image and ask question with your image interactively.

## Usage

1. Clone or download this repository locally. The models is not included, they will be downloaded from Hugging face while running the ap

```bash
git clone https://github.com/thanhtantran/MiniCPM-V-2_6-rkllm
cd MiniCPM-V-2_6-rkllm
```
   
3. The RKNPU2 kernel driver version on the development board must be >=0.9.6 to run such a large model. 
   Use the following command with root privileges to check the driver version:
   ```bash
   > cat /sys/kernel/debug/rknpu/version 
   RKNPU driver: v0.9.8
   ```
   If the version is too low, please update the driver. You may need to update the kernel or refer to official documentation for help.
   
4. Install dependencies, lib ...

```bash
pip install -r requirements.txt
```
You also need to install rknn-toolkit2-lite, this can work with the latest rknn-toolkit2-lite so you can install by

```bash
pip install rknn-toolkit-lite2
```
In case you want to install specific rknn-toolkit-lite2, go to https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2 and download it

This model works with RKNN lib 1.1.4, so if you don't have the lib, or have higher version lib, just replace

```bash
sudo cp lib/* /usr/lib

```

4. Run the app
   
```bash
streamlit run streamlit_app.py
```
Then you can go to your browser to use this app. The address is http://localhost:8501 or http://YOUR_ORANGEPI_IP:8501

5. First step, download model, click button to download models. This will take some minnutes to wait, grab a coffee. You have to do it only 1 time. If your models are already in the machine, you don't need to download them again

![minicpm-demo1](https://github.com/user-attachments/assets/5e893143-3387-4806-87e6-f75f02313296)

After models are ready, you can click Start Inference Process and wait until the Chat Interface apprear.

Forget the alert "Failed to start inference process. Check console for details." I don't know why it appears, and too lazy to fix this.

6. When you see the 💬 Chat Interface, click Upload image to upload your image to chat with

![minicpm-demo2](https://github.com/user-attachments/assets/b8348ce2-f957-45dc-a0fd-8f1ec89efde8)

7. Ask you question and the Response is bellow, along with performance table

![minicpm-demo3](https://github.com/user-attachments/assets/c1a61f09-ca17-4893-adcd-fcd11c6b6a43)

## References

- [sophgo/LLM-TPU models/MiniCPM-V-2_6](https://github.com/sophgo/LLM-TPU/tree/main/models/MiniCPM-V-2_6)
- [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
- [happyme531/MiniCPM-V-2_6-rkllm](https://huggingface.co/happyme531/MiniCPM-V-2_6-rkllm)
