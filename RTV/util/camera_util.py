import cv2
import os
import subprocess


def list_available_cameras(max_devices=10):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cap.release()
            name = get_camera_name_linux(i)
            available.append((i, name))
        else:
            cap.release()
    return available


def get_camera_name_linux(index):
    try:
        device = f"/dev/video{index}"
        result = subprocess.run(["v4l2-ctl", "-d", device, "--info"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in result.stdout.splitlines():
            if "Card type" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Unknown Camera"


def main():
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found.")
        return
    print("Available cameras:")
    for idx, (cam_id, name) in enumerate(cameras):
        print(f"[{cam_id}] {name}")

    selected = int(input("Select camera by ID: "))
    cap = cv2.VideoCapture(selected)

    if not cap.isOpened():
        print("Failed to open the selected camera.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {selected}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
