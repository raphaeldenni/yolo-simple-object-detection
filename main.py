import cv2
from ultralytics import YOLO

MIN_CONFIDENCE = 0.6


def get_colours(class_index):
    """Function to get class' color"""
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]

    color_index = class_index % len(base_colors)

    color = [
        base_colors[color_index][i]
        + increments[color_index][i] * (class_index // len(base_colors)) % 256
        for i in range(3)
    ]

    return tuple(color)


def draw_boxes(frame, result):
    """Draw YOLO result's boxes"""
    classes_names = result.names

    for box in result.boxes:
        predict_confidence = box.conf[0]

        # check if confidence is greater than minimal confidence
        if predict_confidence <= MIN_CONFIDENCE:
            continue

        # Get box coordinates
        [x1, y1, x2, y2] = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get the class
        class_index = int(box.cls[0])
        class_name = classes_names[class_index]

        colour = get_colours(class_index)

        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        cv2.putText(
            frame,
            f"{class_name} {predict_confidence:.2f}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colour,
            2,
        )

    return frame


def main():
    """Script to detect objects and peoples"""
    yolo = YOLO("yolov10s.pt")

    video_cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) & 0xFF != ord("q"):
        # Get frame and ensure it is available
        ret, frame = video_cap.read()

        if not ret:
            continue

        # Get YOLO model results and draw boxes
        results = yolo.track(frame, stream=True)

        for result in results:
            frame = draw_boxes(frame, result)

        # Show the image
        cv2.imshow("frame", frame)

    # Release the video capture and destroy all windows
    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
