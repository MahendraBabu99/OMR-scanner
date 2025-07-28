import cv2
import numpy as np

# Configuration
NUM_QUESTIONS = 5
NUM_CHOICES = 4
BUBBLE_THRESHOLD = 1000  # Minimum fill threshold
ANSWER_KEY = {0: 2, 1: 4, 2: 2, 3: 2, 4: 2}  # Question: Correct Option (0=A, 1=B...)

def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edged = cv2.Canny(blur, 75, 200)
    return image, orig, edged

def find_document_contour(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def reorder_points(pts):
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    new_pts[0] = pts[np.argmin(s)]
    new_pts[3] = pts[np.argmax(s)]
    new_pts[1] = pts[np.argmin(diff)]
    new_pts[2] = pts[np.argmax(diff)]
    return new_pts

def warp_image(image, pts, w=600, h=800):
    reordered = reorder_points(pts)
    dst_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(reordered, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (w, h))
    return warped

def split_boxes(thresh_img):
    rows = np.vsplit(thresh_img, NUM_QUESTIONS)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, NUM_CHOICES)
        for box in cols:
            boxes.append(box)
    return boxes

def get_marked_answers(boxes):
    answers = []
    for q in range(NUM_QUESTIONS):
        row = boxes[q * NUM_CHOICES:(q + 1) * NUM_CHOICES]
        values = [cv2.countNonZero(b) for b in row]
        max_val = max(values)
        if max_val > BUBBLE_THRESHOLD:
            marked = values.index(max_val)
        answers.append(marked)
    return answers

def draw_results(image, answers, correct_answers):
    h, w = image.shape[:2]
    box_height = h // NUM_QUESTIONS
    box_width = w // NUM_CHOICES

    for q in range(NUM_QUESTIONS):
        marked = answers[q]
        correct = correct_answers[q]
        for c in range(NUM_CHOICES):
            x = c * box_width
            y = q * box_height
            center = (x + box_width // 2, y + box_height // 2)
            if c == correct:
                # Correct answer mark (green circle)
                color = (0, 255, 0)
                cv2.circle(image, center, 15, color, 2)
            if c == marked:
                # Student mark: green if correct, red if wrong
                if marked == correct:
                    cv2.circle(image, center, 10, (0, 255, 0), cv2.FILLED)
                else:
                    cv2.circle(image, center, 10, (0, 0, 255), cv2.FILLED)
    return image

def main(image_path=r"C:\Users\chapa mahindra\Downloads\omr.jpg"):
    image, original, edged = load_and_preprocess(image_path)
    doc_cnt = find_document_contour(edged)

    if doc_cnt is None:
        print("OMR sheet not detected!")
        return

    warped_color = warp_image(original, doc_cnt)
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY_INV)

    boxes = split_boxes(thresh)
    answers = get_marked_answers(boxes)

    score = 0
    for q in range(NUM_QUESTIONS):
        if answers[q] == ANSWER_KEY[q]:
            score += 1

    result_image = draw_results(warped_color.copy(), answers, ANSWER_KEY)
    
    print("Student Answers :", answers)
    print("Correct Answers :", list(ANSWER_KEY.values()))
    print("Score: {}/{}".format(score, NUM_QUESTIONS))

    cv2.imshow("OMR Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()