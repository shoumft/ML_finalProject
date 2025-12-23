import cv2
from ultralytics import YOLO

DROIDCAM_URL = "http://172.20.10.1:4747/video"
MODEL_PATH = "best.pt"

price_table = {
    "Cadina": 45,
    "Cadina_steak": 35,
    "CheetoOrange": 35,
    "CheetoYellow": 35,
    "Doritos_BBQ": 35,
    "Doritos_FlaminHotCheese": 35,
    "Doritos_FlaminHotLemon": 35,
    "Doritos_GarlicSteak": 35,
    "Doritos_GoldenCheese": 35,
    "Doritos_garlicPrawn": 35,
    "Doritos_nachoCheese": 35,
    "HwaYuan_squidCrackers": 35,
    "HwaYuan_vegetableCrackers": 35,
    "Karamucho": 35,
    "KaramuchoStrongest": 55,
    "Karamucho_spicySeaweed": 35,
    "Koikeya_BBQCornSnacks": 35,
    "Koikeya_RoastedShrimpAndGarlicCornSnacks": 35,
    "Kuaikuai_green": 30,
    "Kuaikuai_red": 30,
    "Kuakuai_yellow": 30,
    "Lays_Cheese": 35,
    "Lays_KyushuSeaweedFlavor": 35,
    "Lays_RoastedRibs": 35,
    "Lays_SteakFlavor": 35,
    "Lays_deepRidgedChicken": 35,
    "Lays_flaminHot": 35,
    "Lays_garlicSalt": 35,
    "Lays_kelpSalt": 35,
    "Lays_natureSalt": 35,
    "Lays_original": 35,
    "Lays_scallops": 35,
    "Lays_sendaiBeef": 35,
    "LonelyGod": 30,
    "LonelyGod_HotCheese": 30,
    "LuckyStar": 45,
    "OyatsuBabyStar_thin": 39,
    "OyatsuBabyStar_wide": 39,
    "OysterOmelette_normal": 35,
    "OysterOmelette_thin": 35,
    "PeaCrackers_Big": 45,
    "PeaCrackers_Original": 30,
    "PeaCrackers_Spicy": 40,
    "Peacock_FishFlavour": 30,
    "Potata": 30,
    "ShrimpStrips": 32,
    "fishShreds": 50,
}
DEFAULT_PRICE = 0
CONF_THRESHOLD = 0.5

##########################################################################################

def run_checkout():
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print("error! no model")
        return

    cap = cv2.VideoCapture(DROIDCAM_URL)

    if not cap.isOpened():
        print("error! no DroidCam")
        return

    window_name = "Smart Checkout System"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        success, frame = cap.read()
        if not success:
            print("error! no frame")
            continue

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

        total_amount = 0
        detected_items = [] 
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])    
                class_name = model.names[cls_id]
                
                price = price_table.get(class_name, DEFAULT_PRICE)

                total_amount += price
                detected_items.append(f"{class_name} (${price})")

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name}: ${price}"

                (w, h), _ = cv2.getTextSize(label, font, 0.6, 1)

                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    font,
                    0.6,
                    (255, 255, 255),
                    1,
                )

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (300, 150 + len(detected_items) * 30),
            (0, 0, 0),
            -1,
        )
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, "Checkout List", (10, 30), font, 0.8, (0, 255, 255), 2)

        y_offset = 70
        for item in detected_items[:10]:
            cv2.putText(
                frame,
                f"- {item}",
                (10, y_offset),
                font,
                0.6,
                (200, 200, 200),
                1,
            )
            y_offset += 30

        if len(detected_items) > 10:
            cv2.putText(
                frame,
                f"... and {len(detected_items) - 10} more",
                (10, y_offset),
                font,
                0.6,
                (200, 200, 200),
                1,
            )
            y_offset += 30

        cv2.line(frame, (10, y_offset), (280, y_offset), (255, 255, 255), 1)
        y_offset += 40

        total_text = f"Total: ${total_amount}"
        cv2.putText(
            frame,
            total_text,
            (10, y_offset),
            font,
            1.2,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_checkout()
