# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python và PyTorch không được lưu trong thư mục output; Ultralytics `8.4.145` được ghi trong cả ba JSON.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không thay đổi checkpoint, threshold, ảnh mẫu hoặc cấu trúc output. Không train model và không dùng CVAT.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id=468`, `class_name="cab"`, `rank=1`, `score=0.510915`, `taxonomy_name="ImageNet-1K"`.
- Record này mô tả toàn ảnh như thế nào? Model xếp `cab` là lớp phù hợp nhất cho toàn bộ ảnh `traffic`. Record không mô tả một chiếc xe cụ thể vì classification không trả về vị trí hoặc bounding box. Quan sát ảnh cho thấy có nhiều phương tiện, nên một nhãn cấp ảnh không thể hiện đầy đủ tất cả object trong cảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Class list đến từ taxonomy ImageNet-1K dùng để huấn luyện checkpoint. Taxonomy được con người và bộ dữ liệu định nghĩa trước; model chỉ dự đoán trong tập lớp đó, không tự tạo lớp mới.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? `class_id` giúp hệ thống xử lý ổn định, `class_name` giúp con người đọc kết quả, còn `taxonomy_name` xác định ID và tên lớp thuộc hệ phân loại nào. Một ID đứng riêng không đủ ý nghĩa và có thể ánh xạ sang lớp khác trong taxonomy khác.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần nói rõ cách chọn chủ thể chính, có cho phép nhiều nhãn hay không, và cách xử lý ảnh không có một chủ thể nổi trội. Annotator phải theo quy tắc này thay vì chọn theo cảm tính hoặc sao chép top-1 của model.
- Vì sao model score không phải ground truth? `0.510915` là model score dùng để xếp hạng prediction. Ground truth phải do con người tạo hoặc xác nhận theo taxonomy và guideline; prediction có score cao vẫn có thể sai.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `class_name="person"`, `score=0.912625`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`, `bbox_width=113.58`, `bbox_height=279.68`.
- Diễn giải vị trí box bằng lời: Box bao người đứng ở phía bên phải ảnh kitchen. Góc trên-trái của box ở khoảng `(385.33, 69.24)` và góc dưới-phải ở khoảng `(498.92, 348.92)`. Tọa độ dùng pixel, với gốc `(0,0)` ở góc trên-trái ảnh, trục x tăng sang phải và trục y tăng xuống dưới.
- So sánh số prediction ở hai threshold: Thí nghiệm có sẵn trong notebook trên ảnh kitchen cho kết quả `17` prediction tại threshold `0.20`, `11` prediction tại `0.35`, và `6` prediction tại `0.60`.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Hạ threshold giữ lại nhiều prediction hơn, nhờ đó có thể tăng độ bao phủ nhưng cũng làm reviewer phải kiểm tra nhiều box hơn và có thể gặp thêm false positive. Tăng threshold làm output gọn hơn nhưng có thể bỏ các object đúng có score thấp. Threshold chỉ lọc prediction, không phải quy tắc loại bỏ ground truth.
- Đề xuất một quy tắc box chặt: Box phải bám sát bốn biên ngoài cùng của phần object nhìn thấy rõ, không chứa quá nhiều nền và không cắt mất phần nhìn thấy của object. Mỗi object cần một box riêng, kể cả khi nhiều object cùng lớp.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định vẽ box theo phần nhìn thấy hay ước lượng toàn bộ object, mức che khuất tối thiểu để vẫn gán nhãn, và có cần thuộc tính `occluded` hoặc `truncated` hay không. Nếu quy tắc chưa rõ, annotator phải chuyển ca cho reviewer thay vì tự suy đoán phần bị khuất hoặc nằm ngoài ảnh.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `instance_id="kitchen-001"`, `class_name="person"`, `score=0.899318`, `polygon_point_count=348`; sáu điểm đầu là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0]]`.
- Polygon bổ sung chi tiết gì so với box? Bounding box chỉ cho biết hình chữ nhật bao quanh object. Polygon mô tả biên dạng chi tiết hơn của người như đầu, vai, tay và chân, đồng thời loại bớt vùng nền vẫn nằm bên trong box.
- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` phân biệt từng object riêng trong output, kể cả khi có nhiều object cùng class. Nó không phải `class_id`, không phải mã định danh con người và không phải tracking ID dùng để theo dõi object qua nhiều frame.
- Đề xuất một quy tắc biên mask: Polygon phải bám sát biên phần object nhìn thấy, không ăn sang nền hoặc object kế bên, và mỗi instance phải có mask riêng. Annotator cần kiểm tra kỹ các vùng lõm và các phần nhỏ như tay, chân hoặc dụng cụ mảnh.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định có gán phần bị che hay chỉ phần nhìn thấy, cách xử lý biên mờ, lỗ bên trong object và ranh giới giữa các object đang tiếp xúc. Nếu không xác định chắc chắn, annotator cần đánh dấu và chuyển reviewer thay vì tự vẽ phần không quan sát được.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cấp ảnh theo taxonomy và guideline | Ảnh traffic có nhiều loại phương tiện nên việc chọn một lớp chính có thể mơ hồ | Chọn nhãn theo quy tắc chủ thể chính; đánh dấu ca nhiều chủ thể nếu guideline yêu cầu | Kiểm tra nhãn có đúng taxonomy và quy tắc chủ thể chính được áp dụng nhất quán không |
| Phát hiện vật thể | Một class và một `bbox_xyxy` cho mỗi object | Trong ảnh kitchen có người chỉ xuất hiện một phần ở mép trái; một số box có thể chứa nhiều nền hoặc model có thể bỏ sót object | Gán đủ object thuộc phạm vi, vẽ box sát phần nhìn thấy và ghi thuộc tính che khuất/cắt mép theo guideline | Kiểm tra class, box lỏng/chặt, object bị bỏ sót, box trùng và cách xử lý truncation |
| Instance segmentation | Một class, polygon/mask và instance riêng cho mỗi object | Biên người, mặt bàn và các vật thể đang tiếp xúc có thể khó phân biệt | Bám biên object, tách các instance cùng lớp và chuyển vùng mơ hồ cho reviewer | Kiểm tra mask có ăn nền, mất phần object, gộp nhầm instance hoặc xử lý biên không nhất quán không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ dùng ba ảnh COCO công khai đã được notebook cố định và kiểm tra checksum; không đưa ảnh cá nhân, dữ liệu khách hàng, khuôn mặt, biển số hoặc dữ liệu nội bộ lên Colab/GitHub công khai. Luôn giữ `IMAGE_ATTRIBUTION.md` cùng output.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach hoặc giảng viên phụ trách qua kênh hỗ trợ của lớp.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
