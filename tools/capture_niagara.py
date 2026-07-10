"""Launch niagara.exe, wait for it to render Bistro, capture the window, and save PNG."""
import subprocess, sys, time, ctypes
from pathlib import Path

NIAGARA_EXE = r"D:\niagara\build\Release\niagara.exe"
BISTRO_GLTF = r"D:\niagara_bistro\bistro.gltf"
OUTPUT = Path(r"D:\RTXNS\output\bistro_test\niagara_new_gt.png")
WAIT_SEC = 12  # let niagara load + build BLAS + render stable

user32 = ctypes.windll.user32

# Bring window to foreground and capture its client area.
def find_window(title):
    hwnd = user32.FindWindowW(None, title)
    return hwnd

def get_client_rect_screen(hwnd):
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    rect = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = POINT()
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top

def capture_screen_region(x, y, w, h, path):
    import ctypes
    from PIL import Image
    gdi32 = ctypes.windll.gdi32
    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
    gdi32.SelectObject(hdc_mem, hbmp)
    gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, 0x00CC0020)  # SRCCOPY
    # Extract pixels
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                    ("biClrImportant", ctypes.c_uint32)]
    bih = BITMAPINFOHEADER()
    bih.biSize = ctypes.sizeof(bih)
    bih.biWidth = w
    bih.biHeight = -h  # top-down
    bih.biPlanes = 1
    bih.biBitCount = 32
    bih.biCompression = 0  # BI_RGB
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buf, ctypes.byref(bih), 0)
    img = Image.frombuffer("RGBA", (w, h), buf.raw, "raw", "BGRA", 0, 1)
    img.convert("RGB").save(path)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)

def main():
    print(f"Launching {NIAGARA_EXE} {BISTRO_GLTF}")
    proc = subprocess.Popen([NIAGARA_EXE, BISTRO_GLTF],
                            cwd=Path(NIAGARA_EXE).parent)

    # Wait for window to appear
    hwnd = 0
    for _ in range(40):
        time.sleep(0.5)
        hwnd = find_window("niagara")
        if hwnd:
            break
    if not hwnd:
        print("ERROR: niagara window not found")
        proc.kill()
        sys.exit(1)
    print(f"Found niagara window hwnd={hwnd}")

    # Bring to foreground
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    time.sleep(1)

    # Wait for scene load + render stabilize
    print(f"Waiting {WAIT_SEC}s for scene load + render...")
    time.sleep(WAIT_SEC)

    # Capture
    x, y, w, h = get_client_rect_screen(hwnd)
    print(f"Client area: ({x},{y}) {w}x{h}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    capture_screen_region(x, y, w, h, OUTPUT)
    print(f"Saved {OUTPUT}")

    # Close niagara
    print("Closing niagara...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Done.")

if __name__ == "__main__":
    main()
