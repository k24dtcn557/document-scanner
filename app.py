import io
import socket
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from scanner import process_document_image

app = FastAPI(title="Mobile Document Scanner")

# Cho phép CORS để gọi API thuận tiện
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ các file tĩnh (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/scan")
async def scan_document(
    file: UploadFile = File(...),
    mode: str = Form("bw"),
    ratio: str = Form("auto"),
):
    """
    Nhận ảnh tải lên từ camera hoặc bộ sưu tập và trả về ảnh đã scan.
    - mode: 'bw' (đen trắng nét), 'gray' (xám khử bóng), 'color' (màu nắn phẳng)
    - ratio: 'auto' (chuẩn 3D tự động), 'a4' (chuẩn A4 1:1.414), 'letter', 'raw'
    """
    if mode not in ["bw", "gray", "color"]:
        mode = "bw"
    if ratio not in ["auto", "a4", "letter", "raw"]:
        ratio = "auto"

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File ảnh rỗng.")

        # Giải mã ảnh từ mảng bytes vào OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Không thể đọc định dạng ảnh.")

        # Xử lý quét tài liệu qua scanner.py
        result = process_document_image(img, mode=mode, aspect_ratio=ratio)

        # Mã hóa kết quả ra định dạng PNG
        success, encoded_img = cv2.imencode(".png", result)
        if not success:
            raise HTTPException(status_code=500, detail="Lỗi khi mã hóa ảnh kết quả.")

        return Response(
            content=encoded_img.tobytes(),
            media_type="image/png",
            headers={
                "X-Processed-Width": str(result.shape[1]),
                "X-Processed-Height": str(result.shape[0]),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


def get_local_ip():
    """Lấy địa chỉ IP mạng nội bộ (Wi-Fi) để truy cập từ điện thoại."""
    try:
        import subprocess
        output = subprocess.check_output(["ifconfig"], text=True)
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("inet ") and not line.startswith("inet 127."):
                ip = line.split()[1]
                if not ip.startswith("10.8."):  # Bỏ qua VPN nếu có
                    return ip
    except Exception:
        pass
    return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn

    local_ip = get_local_ip()
    port = 8000

    print("=" * 60)
    print("🚀 Document Scanner Web App đang chạy!")
    print(f"👉 Mở trên máy tính:   http://localhost:{port}")
    print(f"📱 Mở trên điện thoại: http://{local_ip}:{port}")
    print("   (Đảm bảo điện thoại và máy tính kết nối cùng mạng Wi-Fi)")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port)
