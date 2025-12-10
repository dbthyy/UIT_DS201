# Deep Learning Coursework
Repository này tổng hợp các bài lab trong môn **DS201: Deep Learning trong Khoa học Dữ liệu**.

## Cấu trúc Thư mục

---

## Lab01 – Neural Networks (MLP)
Triển khai mạng neuron nhiều lớp (1-layer và 3-layer) cho bài toán phân loại MNIST. Bao gồm xây dựng lớp MLP, huấn luyện, đánh giá mô hình.

### Files
- Bai-Thuc-Hanh-1.ipynb  
- MLP_1_Layer.py  
- MLP_3_Layer.py  
- Mnist_Dataset.py  
- main.py  
- checkpoints/  
- data.zip  

---

## Lab02 – Convolutional Neural Networks (CNN)
Áp dụng các kiến trúc CNN kinh điển và hiện đại (LeNet, GoogLeNet, ResNet18) cho phân loại ảnh trên MNIST và VinaFood21. Thực hiện huấn luyện, đánh giá và chạy inference với mô hình CNN tự xây dựng và mô hình pretrained.

### Files
- assignments.ipynb  
- main.py  
- run.ipynb  
- train.py  
- data_utils/  
- models/  
  - LeNet.py  
  - GoogLeNet.py  
  - ResNet18.py  
  - pretrained_resnet.py  
- Mnist_Dataset.py  
- VinaFood21_Dataset.py  

---

## Lab03 – Recurrent Neural Networks (RNN)
Xây dựng và huấn luyện các mô hình RNN (LSTM, GRU, BiLSTM) cho bài toán phân loại phone và VSFC tiếng Việt. Bao gồm tiền xử lý dữ liệu, xây dựng kiến trúc, huấn luyện và đánh giá mô hình.

### Files
- run.ipynb  
- train_phoner.py  
- train_vsfc.py  
- data_utils/  
  - phoner.py  
  - vsfc.py  
- models/  
  - lstm.py  
  - gru.py  
  - bilstm.py  
