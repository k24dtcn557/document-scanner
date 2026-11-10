import cv2
import numpy as np

def order_points(pts):
    # Khởi tạo mảng tọa độ 4 góc: [trái-trên, phải-trên, phải-dưới, trái-dưới]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Trái-trên có tổng x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)] # Phải-dưới có tổng x+y lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Phải-trên có hiệu y-x nhỏ nhất
    rect[3] = pts[np.argmax(diff)] # Trái-dưới có hiệu y-x lớn nhất
    return rect

def get_intersection(line1, line2):
    # Tìm điểm giao cắt của 2 đường thẳng dạng (rho, theta)
    rho1, theta1 = float(line1[0][0]), float(line1[0][1])
    rho2, theta2 = float(line2[0][0]), float(line2[0][1])
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    try:
        sol = np.linalg.solve(A, b).flatten()
        return [int(np.round(sol[0].item())), int(np.round(sol[1].item()))]
    except np.linalg.LinAlgError:
        return None


def get_perspective_aspect_ratio(rect, img_shape):
    """
    Ước lượng tỉ lệ khung hình thực tế (chiều cao / chiều rộng) của tài liệu phẳng trong không gian 3D
    dựa trên 4 góc chiếu phối cảnh (Zhang / Sturm perspective rectification), loại bỏ hiện tượng méo do góc nghiêng.
    """
    (tl, tr, br, bl) = rect
    h_img, w_img = img_shape[:2]
    u0 = w_img / 2.0
    v0 = h_img / 2.0
    f = float(max(w_img, h_img))  # Tiêu cự ước tính tiêu chuẩn của camera điện thoại

    m1 = np.array([tl[0] - u0, tl[1] - v0, f], dtype=np.float64)
    m2 = np.array([tr[0] - u0, tr[1] - v0, f], dtype=np.float64)
    m3 = np.array([br[0] - u0, br[1] - v0, f], dtype=np.float64)
    m4 = np.array([bl[0] - u0, bl[1] - v0, f], dtype=np.float64)

    l12 = np.cross(m1, m2)
    l23 = np.cross(m2, m3)
    l34 = np.cross(m3, m4)
    l41 = np.cross(m4, m1)

    v1 = np.cross(l12, l34)
    v2 = np.cross(l23, l41)

    w_est = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h_est = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    fallback_ratio = h_est / max(w_est, 1.0)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return fallback_ratio

    v1 /= norm_v1
    v2 /= norm_v2
    vl = np.cross(v1, v2)
    norm_vl = np.linalg.norm(vl)
    if norm_vl < 1e-6:
        return fallback_ratio

    vl /= norm_vl

    d1 = np.dot(m1, vl)
    if abs(d1) < 1e-6:
        return fallback_ratio

    k2 = np.dot(m2, vl) / d1
    k3 = np.dot(m3, vl) / d1
    k4 = np.dot(m4, vl) / d1

    if k2 <= 0 or k3 <= 0 or k4 <= 0:
        return fallback_ratio

    P1 = m1
    P2 = m2 / k2
    P3 = m3 / k3
    P4 = m4 / k4

    real_w = (np.linalg.norm(P2 - P1) + np.linalg.norm(P3 - P4)) / 2.0
    real_h = (np.linalg.norm(P4 - P1) + np.linalg.norm(P3 - P2)) / 2.0

    ratio = real_h / max(real_w, 1e-6)

    # Nếu tỉ lệ rất gần chuẩn A4 (1.414), tự động căn chuẩn A4
    if abs(ratio - 1.4142) < 0.08:
        return 1.4142
    # Nếu tỉ lệ rất gần US Letter (1.294)
    elif abs(ratio - 1.2941) < 0.06:
        return 1.2941

    return ratio


def apply_brightness_contrast(img, brightness=100, contrast=100):
    """
    Áp dụng hiệu chỉnh độ sáng và tương phản bằng OpenCV.
    brightness: 50–200 (100 = không đổi), tương ứng CSS brightness()
    contrast:   50–250 (100 = không đổi), tương ứng CSS contrast()
    """
    if brightness == 100 and contrast == 100:
        return img
    # alpha: hệ số tương phản (1.0 = không đổi)
    # beta:  độ lệch độ sáng (0 = không đổi); ánh xạ ±100 → ±80 pixel
    alpha = contrast / 100.0
    beta = (brightness - 100) * 0.8
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return result


def process_document_image(img, mode="bw", aspect_ratio="auto", brightness=100, contrast=100):
    """
    Xử lý quét tài liệu từ ảnh numpy (OpenCV).
    mode: "bw" (đen trắng nét), "gray" (thang xám khử bóng), "color" (màu nắn phối cảnh)
    aspect_ratio: "auto" (chuẩn 3D tự động), "a4" (tỉ lệ A4 1:1.414), "letter" (1:1.294), "raw" (theo pixel gốc)
    brightness: 50–200 (100 = không đổi) — độ sáng
    contrast:   50–250 (100 = không đổi) — tương phản
    """
    if img is None:
        raise ValueError("Ảnh đầu vào không hợp lệ hoặc rỗng.")

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img_resized = cv2.resize(img, (int(img.shape[1] / ratio), 500))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 1. Phát hiện biên giấy bằng Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # 2. Tìm 4 góc tài liệu:
    # Cách 1 (Ưu tiên): Tìm đường bao lớn nhất có 4 đỉnh (Contour + approxPolyDP)
    corners = None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    min_doc_area = (img_resized.shape[0] * img_resized.shape[1]) * 0.1  # Tối thiểu 10% diện tích ảnh
    for c in cnts[:10]:
        area = cv2.contourArea(c)
        if area < min_doc_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = approx.reshape(4, 2).astype("float32")
            break

    # Cách 2 (Dự phòng): Tìm đường thẳng bằng Hough Transform nếu contour không ra 4 góc
    if corners is None:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is not None:
            horizontals = []
            verticals = []
            for line in lines:
                rho, theta = float(line[0][0]), float(line[0][1])
                if rho < 0:
                    rho = -rho
                    theta = theta - np.pi
                if theta < np.pi / 4 or theta > 3 * np.pi / 4 or abs(theta - np.pi) < np.pi / 4:
                    verticals.append((rho, theta))
                else:
                    horizontals.append((rho, theta))

            verticals.sort(key=lambda x: x[0])
            horizontals.sort(key=lambda x: x[0])

            if len(horizontals) >= 2 and len(verticals) >= 2:
                h_lines = [
                    np.array([[horizontals[0][0], horizontals[0][1]]]),
                    np.array([[horizontals[-1][0], horizontals[-1][1]]]),
                ]
                v_lines = [
                    np.array([[verticals[0][0], verticals[0][1]]]),
                    np.array([[verticals[-1][0], verticals[-1][1]]]),
                ]
                hough_corners = []
                for h in h_lines:
                    for v in v_lines:
                        pt = get_intersection(h, v)
                        if pt:
                            hough_corners.append(pt)
                if len(hough_corners) == 4:
                    corners = np.array(hough_corners, dtype="float32")

    # 3. Nắn phối cảnh (Perspective Transform)
    if corners is not None and len(corners) == 4:
        corners = corners * ratio
        rect = order_points(corners)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(np.round(float(np.asarray(widthA).squeeze()))), int(np.round(float(np.asarray(widthB).squeeze()))))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(np.round(float(np.asarray(heightA).squeeze()))), int(np.round(float(np.asarray(heightB).squeeze()))))

        # Hiệu chỉnh tỉ lệ khung hình để loại bỏ biến dạng kéo dài do góc chụp nghiêng
        if aspect_ratio == "a4":
            maxHeight = int(round(maxWidth * 1.4142))
        elif aspect_ratio == "letter":
            maxHeight = int(round(maxWidth * 1.2941))
        elif aspect_ratio == "raw":
            pass  # Giữ nguyên pixel cạnh lớn nhất
        else:
            # "auto": Tự động tính tỉ lệ chuẩn 3D loại bỏ góc nghiêng camera
            calc_ratio = get_perspective_aspect_ratio(rect, orig.shape)
            if calc_ratio >= 1.0:  # Tài liệu khổ dọc
                maxHeight = int(round(maxWidth * calc_ratio))
            else:  # Tài liệu khổ ngang
                maxWidth = int(round(maxHeight / calc_ratio))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    else:
        warped = orig.copy()

    # Nếu chọn chế độ màu gốc
    if mode == "color":
        return warped

    # 4. Khử bóng (Shadow Removal) bằng Flat-Field Division
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h_w, w_w = warped_gray.shape

    # Ước lượng nền qua ảnh thu nhỏ để loại bỏ hoàn toàn chữ, chỉ giữ lại ánh sáng & bóng
    scale = 8
    bg_small = cv2.resize(warped_gray, (max(w_w // scale, 16), max(h_w // scale, 16)))
    bg_small = cv2.dilate(bg_small, np.ones((7, 7), np.uint8))
    bg_small = cv2.medianBlur(bg_small, 21)
    bg = cv2.resize(bg_small, (w_w, h_w))

    # Chia nền: Giấy trắng được đưa về màu trắng đồng nhất (~255), chữ giữ nguyên độ tương phản
    flat = cv2.divide(warped_gray, bg, scale=255)

    if mode == "gray":
        # Thang xám mềm mại: Kéo dãn tương phản nhẹ nhàng, nét chữ mượt không bị vỡ hạt
        p_low = np.percentile(flat, 1)
        p_high = np.percentile(flat, 92)
        clean_gray = np.clip((flat.astype(float) - p_low) * (255.0 / max(p_high - p_low, 1.0)), 0, 255).astype(np.uint8)
        return clean_gray

    # 5. Đen trắng (Adaptive Thresholding) dịu nhẹ, không gắt:
    # - Kích thước vùng (block_size) tự co giãn theo độ phân giải ảnh (tránh hạt nhiễu li ti)
    # - Hằng số C được nâng lên 10-12 (chỉ những nét chữ thực sự tối mới thành đen)
    block_size = int(min(h_w, w_w) * 0.02)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(25, block_size)

    # Lọc nhẹ để triệt tiêu các chấm nhiễu trên bề mặt giấy trước khi binarize
    smoothed = cv2.GaussianBlur(flat, (3, 3), 0)
    bw = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10)
    return bw

def process_document(image_path, mode="bw", aspect_ratio="auto"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy hoặc không đọc được ảnh: {image_path}")
    return process_document_image(img, mode=mode, aspect_ratio=aspect_ratio)

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Document scanner using OpenCV")
    parser.add_argument("image", help="Đường dẫn tới file ảnh cần scan")
    parser.add_argument("-o", "--output", default="output.png", help="Đường dẫn lưu kết quả (mặc định: output.png)")
    parser.add_argument("-m", "--mode", choices=["bw", "gray", "color"], default="bw", help="Chế độ quét: bw (đen trắng), gray (xám), color (màu)")
    parser.add_argument("-r", "--ratio", choices=["auto", "a4", "letter", "raw"], default="auto", help="Tỉ lệ khung hình: auto (tự động 3D), a4 (chuẩn A4), letter (chuẩn Letter), raw (kích thước pixel)")
    parser.add_argument("--show", action="store_true", help="Hiển thị ảnh kết quả lên màn hình")

    args = parser.parse_args()

    try:
        scanned = process_document(args.image, mode=args.mode, aspect_ratio=args.ratio)
        cv2.imwrite(args.output, scanned)
        print(f"✅ Đã scan thành công và lưu tại: {args.output}")

        if args.show:
            cv2.imshow("Scanned Document", scanned)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ Lỗi: {e}", file=sys.stderr)
        sys.exit(1)
