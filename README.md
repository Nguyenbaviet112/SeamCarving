# Đồ án cuối kỳ

**Song song hóa và tối ưu hóa một ứng dụng** :

### Thành viên nhóm 

  *  21424088 - Huỳnh Văn Thái
  *  21424069 - Nguyễn Bá Việt
  *  21424032 - Võ Công Minh


### Tên dự án

Song song hóa và tối ưu hóa ứng dụng Thay đổi kích thước ảnh bằng thuật toán Seam carving.

### Vấn đề

Để triển khai Thuật toán Seam Carving xử lý hình ảnh bằng cách sử dụng phương pháp tiếp cận tuần tự và song song. Việc triển khai tuần tự sẽ được thực hiện trên host CPU (Bộ xử lý trung tâm) và việc triển khai song song sẽ được thực hiện trên GPU NVidia hỗ trợ CUDA (Kiến trúc thiết bị điện toán hợp nhất).

_Mục đích chính của phần trình diễn này là cho thấy sự khác biệt giữa tốc độ tính toán của CPU và GPU._

# I. Giới thiệu

### 1. Seam Carving là gì?

Theo Wikipedia :

> **Seam carving is an algorithm for image resizing**, developed by Shai Avidan, of Mitsubishi
> Electric Research Laboratories (MERL), and Ariel Shamir, of the Interdisciplinary Center and MERL. It
> functions by establishing a number of seams (paths of least importance) in an image and **automatically
> removes seams to reduce image size or inserts seams to extend it.** Seam carving also allows manually
> defining areas in which pixels may not be modified, and **features the ability to remove whole objects from
> photographs.**
>
> _The purpose of the algorithm is to display images without distortion on various media (cell
> phones, PDAs) using document standards, like HTML, that already support dynamic changes in page
> layout and text, but not images._

* *Seam carving là một thuật toán để thay đổi kích thước hình ảnh, tự động loại bỏ các đường seam để giảm kích thước hình ảnh hoặc chèn các seam để mở rộng hình ảnh*

* *Hãy xem [video giải thích](https://youtu.be/6NcIJXTlugc) này để hiểu thêm chi tiết.*


![image](https://drive.google.com/uc?id=1MU6FCaEDSWlqX8osiN5oRoKe76gBOV6s)

### 2. Một Seam là gì?

Theo Wikipedia :

>Seams can be either vertical or horizontal. **A vertical seam is a path of connected pixels from top
to bottom in an image with one pixel in each row.** A horizontal seam is similar with the exception of the
connection being from left to right. The importance/energy function values a pixel by measuring its
contrast with its neighbor pixels.

* *Một seam: một tập các pixel, mỗi dòng một pixel, pixel của dòng r & dòng r+1 được kết nối với nhau.*


## II. Nội dung
### 1. Mô tả dự án

- Input: một ảnh đầu vào RGB `in.pnm`

![image](https://drive.google.com/uc?id=154_BR4vkQW6cnE38xxpzF8VAu9p9D_pd)

- Output: một ảnh đầu ra RGB `out.pnm` được thay đổi kích thước **mà không làm biến dạng các đối tượng quan trọng** (tấm ảnh được thu hẹp chiều rộng lại).

![image](https://drive.google.com/uc?id=1KI3J_aa9SNJDOAAoANj-PeuXDwdp_LFu)

- File khung chương trình `seamcarving_v1.cu`.

### 2. Ý nghĩa thực tế:

  - Ứng dụng seam carving được sử dụng để thay đổi kích thước hình ảnh mà không làm biến đổi hoặc làm thay đổi nội dung của hình ảnh ban đầu. Thay vì cắt hoặc co giãn hình ảnh, seam carving tìm kiếm các đường seam (dòng điểm ảnh) ít quan trọng và xóa chúng hoặc chèn thêm chúng, điều này cho phép thay đổi kích thước hình ảnh một cách không đáng kể mà vẫn giữ được các đặc điểm quan trọng của hình ảnh ban đầu.

  - Ứng dụng của seam carving là rất đa dạng, từ việc tạo ra các ảnh thu nhỏ hoặc phóng to cho các trang web hay ứng dụng di động, đến việc chỉnh sửa kích thước hình ảnh trong các dự án thiết kế đồ họa và phim ảnh. Nó cũng có thể được sử dụng để loại bỏ các đối tượng không mong muốn trong hình ảnh.

- Ứng dụng này có cần tăng tốc không?

  - Tùy thuộc vào kích thước và độ phức tạp của hình ảnh, quá trình seam carving có thể mất nhiều thời gian để hoàn thành. Vì vậy, Việc tăng tốc xử lý là cần thiết để đảm bảo rằng quá trình seam carving diễn ra nhanh chóng và hiệu quả.

  - Phương pháp tối ưu hóa được sử dụng để cải thiện tốc độ xử lý trong seam carving được đề xuất trong dự án này : **Sử dụng phần cứng tăng tốc như GPU để giảm thời gian xử lý**.



### III. Cài đặt tuần tự 


#### 1. Ý tưởng

Thực hiện tuần từ qua các bước:

- **Bước 1. Khởi tạo và chuẩn bị dữ liệu:** Hàm khởi tạo và chuẩn bị các biến và mảng cần thiết cho việc Seam Carving, bao gồm mảng `inGrayscale` (lưu trữ ảnh xám đầu vào), mảng `energy` (lưu trữ mức độ quan trọng của mỗi pixel), ma trận `cost_v` (ma trận chi phí theo chiều dọc), mảng `importancy_h` (lưu trữ độ quan trọng theo chiều ngang), mảng `next_pixels_v` (lưu trữ chỉ số của Seam nhỏ nhất theo chiều dọc), và mảng `next_pixels_h` (lưu trữ chỉ số của Seam nhỏ nhất theo chiều ngang).

- **Bước 2. Chuyển đổi từ RGB sang Ảnh xám:** Bước này bao gồm chuyển đổi ảnh đầu vào từ không gian màu RGB sang ảnh xám. Mỗi giá trị RGB của pixel được chuyển đổi thành một giá trị xám duy nhất sử dụng công thức: 

> Y = 0.299R + 0.587G + 0.114B

Các giá trị pixel xám kết quả được lưu trữ trong mảng **inGrayscale**.

- **Bước 3. Tính toán mức độ quan trọng của từng pixel (dùng edge detection):** Trong bước này, mức độ quan trọng của mỗi pixel được tính toán dựa trên phát hiện biên. Các bộ lọc phát hiện biên (như **Sobel**) được áp dụng lên ảnh xám để tính toán đạo hàm theo chiều ngang và chiều dọc. Sau đó, mức độ quan trọng của mỗi pixel được tính toán bằng cách tổng hợp giá trị tuyệt đối của đạo hàm theo cả hai chiều. Kết quả được lưu trữ trong mảng `energy`.

> - Phát hiện cạnh theo chiều x (1): thực hiện convolution giữa ảnh grayscale với bộ lọc x-Sobel
>
> - Phát hiện cạnh theo chiều y (2): thực hiện convolution giữa ảnh grayscale với bộ lọc y-Sobel
>
> - Độ quan trọng của một pixel = |kết quả tương ứng của (1)| + |kết quả tương ứng của (2)|

![image](https://drive.google.com/uc?id=17_PF-IO4FK60l0BU5-05Lma43c0wUjah)

- **Bước 4. Tính ma trận chi phí theo chiều dọc:** Hàm tính toán ma trận chi phí theo chiều dọc của ảnh. Ma trận `cost_v` có cùng kích thước với ảnh xám và được khởi tạo bằng các giá trị mức độ quan trọng từ mảng `energy`. Tiếp theo, hàm duyệt qua từng hàng của ma trận `cost_v` từ trên xuống dưới và từ trái sang phải. Tại mỗi vị trí, chi phí của pixel được tính toán bằng cách xem xét đường đi có chi phí nhỏ nhất từ hàng bên dưới. Chi phí của mỗi pixel trong ma trận `cost_v` được cập nhật dựa trên chi phí tính toán và mức độ quan trọng của pixel tương ứng.

> giá trị cost_v[i][j] được tính như sau:
>
> Nếu j là cột đầu tiên, lấy giá trị bên phải của cost_v[i-1][j]
>
> Nếu j là cột cuối cùng, lấy giá trị bên trái của cost_v[i-1][j]
>
> Nếu j không nằm ở cột đầu tiên hoặc cuối cùng, lấy giá trị nhỏ nhất của hai phần tử bên trái và bên phải của nó trong hàng trên cùng của ma trận cost_v rồi cộng với energy[i][j]

![image](https://drive.google.com/uc?id=1UiGRuGpjnDOPHq70Ihcbf32b2W4h4O_U)

- **Bước 5. Tìm Seam nhỏ nhất:** Hàm tìm Seam nhỏ nhất bằng cách xác định đường đi có tổng chi phí nhỏ nhất từ đỉnh đến đáy của ma trận `cost_v`. Quá trình này được thực hiện theo các bước sau:

   - Đầu tiên, Seam nhỏ nhất được khởi tạo bằng cách tìm pixel có chi phí nhỏ nhất ở hàng cuối cùng của ma trận `cost_v`. Giá trị của pixel này được ghi lại vào mảng `next_pixels_v` tại vị trí tương ứng.

   - Tiếp theo, hàm duyệt qua từng hàng từ hàng cuối cùng trở lên hàng đầu tiên. Tại mỗi hàng, vị trí của Seam nhỏ nhất được xác định dựa trên giá trị của pixel ở hàng dưới. Hàm kiểm tra 3 vị trí trên hàng dưới, tương ứng với pixel trước, pixel hiện tại và pixel sau, và chọn vị trí có chi phí nhỏ nhất. Giá trị của pixel được chọn là giá trị tại vị trí đó trên hàng dưới cộng với mức độ quan trọng của pixel hiện tại. Vị trí của Seam nhỏ nhất được ghi lại vào mảng `next_pixels_v` tại vị trí tương ứng.

   - Sau khi hoàn thành vòng lặp, mảng `next_pixels_v` sẽ chứa chỉ số của Seam nhỏ nhất từ đỉnh đến đáy của ma trận `cost_v`.

  Quá trình tìm Seam nhỏ nhất này xảy ra theo chiều dọc, tạo ra một đường đi từ đỉnh đến đáy của ảnh.

  ![image](https://drive.google.com/uc?id=1x5xTymCw-v1rG7ULmftweBE1Po2NGL1h)

- **Bước 6. Xóa Seam nhỏ nhất:** Bước này nhằm loại bỏ Seam nhỏ nhất khỏi ảnh ban đầu. Quá trình này được thực hiện như sau:

  - Một vòng lặp duyệt qua từng hàng của ảnh được thực hiện. Trong mỗi hàng, ta duyệt qua từng pixel trên hàng đó.

  - Nếu pixel hiện tại nằm trước vị trí Seam nhỏ nhất (tức là có chỉ số nhỏ hơn), ta giữ nguyên giá trị của pixel đó và sao chép nó vào ảnh kết quả.

  - Ngược lại, nếu pixel hiện tại nằm sau vị trí Seam nhỏ nhất, ta dịch chuyển chỉ số của pixel hiện tại sang trái một vị trí (bỏ qua Seam nhỏ nhất). Sau đó, ta sao chép giá trị của pixel sau dịch chuyển vào ảnh kết quả.
  - Quá trình trên được lặp lại cho tất cả các hàng của ảnh.

  Sau khi hoàn thành vòng lặp, chiều rộng của ảnh sẽ được giảm đi một đơn vị do Seam nhỏ nhất đã được loại bỏ. Ảnh kết quả sẽ được lưu vào mảng `outPixels`.

- **Bước 7. Ghi ảnh kết quả:** Bước này nhằm ghi lại ảnh kết quả sau khi đã xóa Seam nhỏ nhất từ ảnh ban đầu.

#### 2. Tiến hành

Tiến hành chạy chương trình SeamCarving tuần tự trên:

- Đầu vào là ảnh có kích thước 640x434 tạo ra ba ảnh có chiều rộng lần lượt là 400, 300, 200