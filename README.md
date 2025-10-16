## 🎥 Demo
![Demo](demo.gif)

# YOLOv1 — You Only Look Once (2016)

YOLOv1 là phiên bản đầu tiên của họ mô hình **YOLO (You Only Look Once)** — một hướng tiếp cận **phát hiện vật thể (object detection)** hoàn toàn mới, được đề xuất bởi **Joseph Redmon, Santosh Divvala, Ross Girshick và Ali Farhadi** năm 2016.


## 1. Ý tưởng cốt lõi

YOLOv1 chia ảnh đầu vào thành một lưới (grid) có kích thước $S \times S$ (thường là $7 \times 7$).

Mỗi cell trong grid sẽ:

* **Dự đoán B bounding boxes** (thường $B=2$).
* **Confidence score** cho mỗi box (độ tin cậy có vật thể trong box).
* **Class probabilities** (xác suất thuộc từng lớp vật thể).

Nếu tâm của vật thể nằm trong một cell, cell đó sẽ chịu trách nhiệm phát hiện vật thể đó.



##  2. Kiến trúc mạng YOLOv1
![](https://images.viblo.asia/58986feb-d8ac-4255-aee5-329f41bf801d.png)
YOLOv1 sử dụng một **CNN gồm 24 lớp convolutional** và **2 lớp fully connected**, hoạt động tương tự như **GoogLeNet** (không có Inception).

Cấu trúc tổng quan:

| Loại layer            | Số lớp | Mục đích                                  |
| --------------------- | ------ | ----------------------------------------- |
| Conv + ReLU + MaxPool | 24     | Trích xuất đặc trưng (feature extraction) |
| Fully Connected       | 2      | Dự đoán bounding box và class             |

* Kích thước ảnh đầu vào: $$448 \times 448 \times 3$$
* Đầu ra cuối cùng: $$7 \times 7 \times (B \times 5 + C)$$

Với:

* $$B=2$$ (2 bounding boxes mỗi cell)
* $$C=20$$ (20 lớp trong Pascal VOC)

→ Kích thước đầu ra:
$$7 \times 7 \times (2 \times 5 + 20) = 7 \times 7 \times 30$$


##  3. Biểu diễn đầu ra

Mỗi cell dự đoán:

* **2 bounding boxes**, mỗi box gồm 5 giá trị:
  $$(x, y, w, h, C)$$
  Trong đó:

  * $x, y$: tọa độ tâm box, **chuẩn hóa** theo cell.
  * $w, h$: chiều rộng và cao, **chuẩn hóa theo toàn ảnh**.
  * $C$: confidence score (độ tin cậy của box).

**Confidence score** được định nghĩa là:

$$
C = P(\text{object}) \times \text{IoU}_{\text{pred, truth}}
$$

Trong đó:

* $$P(\text{object})$$ là xác suất có vật thể trong cell.
* $$\text{IoU}_{\text{pred, truth}}$$ là giao trên hợp giữa box dự đoán và box thật.

Ngoài ra, mỗi cell còn dự đoán:

* **20 class probabilities**:
  $$P(\text{class}_i | \text{object})$$


## 4. Pipeline xử lý

1. **Chia ảnh đầu vào** thành $7 \times 7$ grid.
2. **Mỗi cell** dự đoán:

   * 2 bounding boxes
   * 1 confidence score cho mỗi box
   * 20 class probabilities
3. **Tính toán class-specific confidence score**:
   $$
   P(\text{class}*i) \times \text{IoU}*{\text{pred, truth}}
   $$
4. **Áp dụng Non-Max Suppression (NMS)** để loại bỏ các box trùng lặp.

## 5. Hàm mất mát (Loss function)

YOLOv1 sử dụng một loss function duy nhất để huấn luyện toàn bộ mô hình. Loss này là tổng bình phương sai số (sum-squared error) giữa giá trị dự đoán và ground truth.

Tổng quát, loss bao gồm 3 phần chính:

$$
\text{Loss} = \text{Loss}*{\text{coord}} + \text{Loss}*{\text{confidence}} + \text{Loss}_{\text{class}}
$$

### 1. Localization Loss (tọa độ box)

$$
\lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}}
\left[
(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2
\right]
$$

→ Mục tiêu: Dự đoán chính xác tọa độ và kích thước box.

### 2. Confidence Loss

* Với box **có vật thể**:
  $$
  \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2
  $$
* Với box **không có vật thể**:
  $$
  \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2
  $$

### 3. Classification Loss

$$
\sum_{i=0}^{S^2} \mathbb{1}*i^{\text{obj}} \sum*{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$

###  Hệ số điều chỉnh:

* $\lambda_{\text{coord}} = 5$ — tăng trọng số cho phần localization.
* $\lambda_{\text{noobj}} = 0.5$ — giảm ảnh hưởng của các cell không có vật thể.



##  6. Kết quả và hiệu năng

* **Dataset**: Pascal VOC 2007/2012
* **Tốc độ**: ~45 FPS (phiên bản full YOLO)
* **Tốc độ nhanh (Fast YOLO)**: ~155 FPS
* **mAP**: ~63.4% trên VOC 2007



##  7. Hạn chế của YOLOv1

1. **Không tốt với vật thể nhỏ**
   → Vì chia lưới 7×7, một cell chỉ dự đoán 1 vật thể.

2. **Không phát hiện tốt các vật thể gần nhau**
   → Nếu 2 vật thể nằm trong cùng 1 cell, YOLO chỉ dự đoán được 1.

3. **Localization chưa chính xác**
   → Dự đoán bounding box còn sai lệch ở góc hoặc tỷ lệ.

4. **Tổng hợp loss không cân bằng**
   → Dễ bị chi phối bởi lỗi confidence.


#  YOLOv2 



##  1. Tổng quan ý tưởng

YOLOv2 khắc phục nhiều hạn chế của YOLOv1:

* Thêm **anchor boxes** giống Faster R-CNN.
* Áp dụng **batch normalization**, **high-resolution classifier**, **multi-scale training**.
* Tích hợp **WordTree** để huấn luyện chung 2 dataset (VOC + ImageNet).




YOLOv2 giới thiệu backbone mới: **Darknet-19**
(19 convolutional layers + 5 maxpool layers).

| Loại layer      | Số lượng | Kích thước kernel                  | Ghi chú                      |
| --------------- | -------- | ---------------------------------- | ---------------------------- |
| Convolution     | 19       | $$1 \times 1$$ hoặc $$3 \times 3$$ | Có BatchNorm + LeakyReLU     |
| MaxPooling      | 5        | $$2 \times 2$$                     | Giảm kích thước feature map  |
| Fully Connected | 0        | —                                  | Không còn dùng FC như YOLOv1 |

* Input: $$416 \times 416$$
* Output feature map: $$13 \times 13$$


##  3. Các cải tiến chính so với YOLOv1

### 1. Batch Normalization (BN)

Thêm **BatchNorm** vào mọi convolution layer giúp:

* Tăng độ ổn định khi huấn luyện.
* Loại bỏ nhu cầu sử dụng dropout.
* Tăng mAP ~2%.

$$
\text{BN}(x) = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$


### 2. High-Resolution Classifier

Trong YOLOv1, mạng được huấn luyện phân loại ảnh kích thước $$224 \times 224$$ (giống ImageNet).
YOLOv2 thay đổi thành $$448 \times 448$$ **trước khi huấn luyện detection**, giúp mạng **thích nghi với ảnh độ phân giải cao hơn**.



### 3. Anchor Boxes

YOLOv2 **chuyển từ trực tiếp dự đoán $(x, y, w, h)$ sang sử dụng anchor boxes** giống như Faster R-CNN và SSD.

Giờ đây, mỗi cell dự đoán **n anchor boxes** (thường là 5), giúp mô hình:

* Phát hiện **nhiều vật thể** trong 1 cell.
* **Ổn định hơn** trong huấn luyện.

Cách xác định anchor box:

* Dùng **K-means clustering** trên ground truth boxes để tìm 5 anchor tối ưu.
* Khoảng cách được đo bằng:
  $$
  d(\text{box}_1, \text{box}_2) = 1 - \text{IoU}(\text{box}_1, \text{box}_2)
  $$

Cách tính toán bounding box dự đoán:

$$
\begin{aligned}
b_x &= \sigma(t_x) + c_x \
b_y &= \sigma(t_y) + c_y \
b_w &= p_w \cdot e^{t_w} \
b_h &= p_h \cdot e^{t_h}
\end{aligned}
$$

Trong đó:

* $(c_x, c_y)$ là tọa độ cell trong grid.
* $(p_w, p_h)$ là kích thước anchor box.
* $(t_x, t_y, t_w, t_h)$ là giá trị mạng dự đoán.
* $\sigma$ là hàm sigmoid đảm bảo $b_x, b_y$ trong [0, 1].



### 4. Dimension Clusters

YOLOv2 **tự động chọn anchor box kích thước tối ưu** bằng K-means, thay vì đặt thủ công như Faster R-CNN.

→ Các anchor phản ánh **phân bố thực tế** của kích thước vật thể trong tập huấn luyện.



### 5. Fine-grained Features

YOLOv2 bổ sung **skip connection** (giống ResNet) từ layer trung gian sang feature map cuối.

Điều này giúp mô hình:

* Giữ lại thông tin chi tiết về **vị trí (spatial)**.
* Phát hiện vật thể nhỏ tốt hơn.


### 6. Multi-Scale Training

Mỗi 10 batch, YOLOv2 **thay đổi kích thước đầu vào** ngẫu nhiên trong khoảng 320 → 608 (bội số của 32).

→ Giúp mạng **mạnh mẽ hơn với nhiều độ phân giải**,
→ Có thể chạy nhanh hoặc chính xác tùy tình huống.



### 7. WordTree + Hierarchical Classification

YOLOv2 được huấn luyện trên **2 dataset song song**:

* Pascal VOC (20 lớp có bounding box)
* ImageNet (9000 lớp chỉ có label)

Bằng cách kết hợp chúng qua cấu trúc WordTree (dựa trên WordNet), mô hình học được **quan hệ phân cấp giữa các lớp**.

Ví dụ:


Animal → Dog → German Shepherd


Khi đó, nếu ảnh là “German Shepherd” nhưng YOLO chỉ đoán “Dog”, mô hình vẫn được xem là đúng một phần.
