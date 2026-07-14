# VisageAI — AI Face Rating Pipeline

> Hệ thống đánh giá thẩm mỹ khuôn mặt sử dụng **MediaPipe Face Mesh** và **DINOv2**.  
> Kết hợp đặc trưng hình học học thuật (geometric features) với đặc trưng học sâu (vision foundation model) để tạo ra một biểu diễn đặc trưng toàn diện cho khuôn mặt.

---

## Tính năng chính (v0.3)

### Geometric Features (49 chiều)
| Nhóm | Chỉ số |
|---|---|
| **Tỷ lệ khuôn mặt** | fWHR, Face Aspect Ratio, Midface Ratio, Facial Thirds, Facial Fifths |
| **Mắt** | Canthal Tilt, Eye Spacing Ratio, Eye Openness, Interpupillary Distance, Intercanthal Distance |
| **Mũi** | Nose Width Ratio, Nose Length Ratio |
| **Môi** | Philtrum Ratio, Lip Thickness Ratio |
| **Quai hàm** | Jaw Width, Jaw Angle (polyfit), Jaw Curvature, Chin Angle, Cheekbone Width |
| **Lông mày** | Eyebrow Height |
| **Đối xứng** | Landmark Symmetry Score |

### Deep Learning Features (384 chiều)
- **DINOv2 ViT-S/14** — CLS token và/hoặc Mean Pool patch tokens
- Trích xuất trực tiếp qua `torch.hub`, chạy trên CUDA

### Quality Assessment
- Head Pose (Yaw/Pitch/Roll) via `solvePnP`
- Blur score (Laplacian variance)
- **Geometry gate**: tự động bỏ qua geometric metrics khi `|yaw| > 15°`

### Feature Fusion
- **433-d vector** = 49 geo + 384 DINOv2 CLS
- Baseline v0.3: simple concatenation
- MLP regression head sẽ được huấn luyện ở v0.4 với dữ liệu từ SCUT-FBP5500

---

## Kiến trúc Pipeline

```
Input Image
     │
     ▼
Face Detection & Alignment (MediaPipe Face Mesh)
     │
     ├──► Quality Assessment (Head Pose, Blur, Resolution)
     │         │
     │         └──► geometry_valid flag
     │
     ├──► Geometric Feature Extraction (49 features)
     │         ├── Jaw Width / Chin Angle / Jaw Curvature
     │         ├── fWHR / Face Aspect Ratio / Midface Ratio
     │         ├── Facial Thirds / Facial Fifths
     │         ├── Eye Openness / Canthal Tilt / IPD / ICD
     │         ├── Nose Width & Length Ratios
     │         ├── Lip Thickness / Philtrum Ratio
     │         ├── Cheekbone Width / Eyebrow Height
     │         └── Landmark Symmetry Score
     │
     ├──► DINOv2 Feature Extraction (384-d CLS / Mean Pool / Both)
     │
     └──► Feature Fusion (433-d vector)
               └──► JSON Report + Annotated Image
```

---

## Cài đặt

### Yêu cầu
- Python 3.10+
- CUDA (khuyến nghị, fallback CPU)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý phiên bản quan trọng:**  
> MediaPipe 0.10.20 yêu cầu `numpy < 2` và `protobuf 4.x`.  
> `requirements.txt` đã pin các phiên bản tương thích.

---

## Sử dụng

```bash
python src/main.py \
    --image path/to/face.jpg \
    --output_dir output \
    --pool both        # cls | mean | both
```

### Tham số

| Tham số | Mô tả | Mặc định |
|---|---|---|
| `--image` | Đường dẫn đến ảnh đầu vào | *(bắt buộc)* |
| `--output_dir` | Thư mục lưu kết quả | `output` |
| `--model` | Tên mô hình DINOv2 | `dinov2_vits14` |
| `--pool` | Chiến lược pooling: `cls`, `mean`, `both` | `both` |
| `--skip_dl` | Bỏ qua DINOv2 (chỉ tính geometry) | `False` |

### Đầu ra

```
output/
├── <name>_aligned.jpg      # Ảnh đã căn chỉnh thẳng
├── <name>_annotated.jpg    # Ảnh với HUD đo đạc hình học
└── <name>_report.json      # Toàn bộ số liệu dưới dạng JSON
```

### Ví dụ JSON output (trích)

```json
{
  "pipeline_version": "v0.3",
  "face_quality": {
    "face_detected": true,
    "alignment_angle_deg": -0.03,
    "head_pose": { "yaw": -0.46, "pitch": -10.12, "roll": -169.65 },
    "blur": { "score": 273.52, "is_sharp": true },
    "geometry_valid": true
  },
  "geometric_metrics": {
    "fwhr": { "value": 1.64, "width_landmarks": "234-454 (bizygomatic approx)" },
    "canthal_tilt": { "left_deg": 4.67, "right_deg": 4.15 },
    "jaw_curvature": { "curvature_ratio": 1.88, "note": "1.0=square, higher=V-shaped" },
    "interpupillary_distance": { "ipd_px": 155.4, "method": "iris_center (landmarks 468, 473)" },
    "landmark_symmetry_score": 0.9438
  },
  "fused_feature": {
    "total_dims": 433,
    "geo_dims": 49,
    "dino_dims": 384
  }
}
```

---

## Cấu trúc Project

```
VisageAI/
├── requirements.txt
└── src/
    ├── main.py                   # Điều phối pipeline chính
    ├── features/
    │   ├── landmarks.py          # MediaPipe Face Mesh extractor (478-point)
    │   ├── geometry.py           # 20+ geometric feature functions + references
    │   ├── quality.py            # Head pose (solvePnP), blur, geometry gate
    │   └── fusion.py             # geometry_to_vector + fuse_features
    ├── models/
    │   └── backbone.py           # DINOv2 extractor (cls / mean / both pooling)
    └── utils/
        └── visualization.py      # HUD overlay, facial annotations
```

---

## Tài liệu tham khảo (Scientific References)

Mỗi chỉ số hình học trong `geometry.py` có dẫn chiếu cụ thể:

- Farkas (1994) *Anthropometry of the Head and Face*
- Carré et al. (2008) *Measuring the face: Facial width-to-height ratio*
- Lefevre et al. (2012) *New method for measuring facial width-to-height ratio*
- Naini et al. (2012) *Facial Aesthetics: Concepts and Clinical Diagnosis*
- Baudouin & Tiberghien (2004) *Symmetry, averageness, and feature size in facial attractiveness*
- Price et al. (2011) *Palpebral fissure dimensions*
- Dodgson (2004) *Variation and Extrema of Human Interpupillary Distance*
- Oquab et al. (2023) *DINOv2: Learning Robust Visual Features without Supervision*
- El Banani et al. (2023) *Probing the 3D Awareness of Visual Foundation Models*

---

## Roadmap

| Phiên bản | Nội dung |
|---|---|
| **v0.1** | ✅ Pipeline baseline: landmarks + DINOv2 + visualization |
| **v0.2** | ✅ Quality gate, Feature Fusion, Facial Fifths, Jaw polyfit |
| **v0.3** | ✅ 16 feature groups mới (Jaw Width, Chin Angle, Eye Openness, IPD, ...) |
| **v0.4** | 🔜 Dataset integration (SCUT-FBP5500), MLP regression head |
| **v0.5** | 🔜 Multi-task training (overall score + attribute scores) |
| **v1.0** | 🔜 Web interface (FastAPI + React) |

---

## Lưu ý / Disclaimer

Dự án này được xây dựng cho mục đích **nghiên cứu học thuật** về Computer Vision và Facial Analysis.  
Các số liệu thẩm mỹ trong dự án này **không đại diện cho một chuẩn đẹp tuyệt đối** và không nên được dùng để đánh giá hay so sánh con người ngoài bối cảnh học thuật.
