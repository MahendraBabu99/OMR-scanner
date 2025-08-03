
import cv2
import numpy as np
import random

# Generate a random answer key of size 160 with options A, B, C, D
random_answer_key = [random.choice(['A', 'B', 'C', 'D']) for _ in range(160)]


# For 160 questions, 5 columns, 32 questions per column
def group_questions_column_wise_row_bubbles(cnts, questions_per_col=32, options_per_question=4, col_tolerance=50, row_tolerance=30):
    # Sort left to right (x-axis) for columns
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
    columns = []
    current_col = []
    last_x = None

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if last_x is None or abs(x - last_x) < col_tolerance:
            current_col.append(c)
        else:
            columns.append(current_col)
            current_col = [c]
        last_x = x
    if current_col:
        columns.append(current_col)

    all_questions = []
    for col in columns:
        # sort top to bottom within column
        col = sorted(col, key=lambda c: cv2.boundingRect(c)[1])
        for i in range(0, len(col), options_per_question):
            group = col[i:i + options_per_question]
            if len(group) == options_per_question:
                # sort options left to right (row-wise within question)
                group = sorted(group, key=lambda c: cv2.boundingRect(c)[0])
                all_questions.append(group)

    return all_questions


# --- Main Program ---
image = cv2.imread(r"C:\Users\chapa mahindra\Documents\varshitha.jpg")
if image is None:
    print("Image not found.")
    exit()

# Resize for easier processing (optional, depends on image size)
resized_image = cv2.resize(image, (1200, 1700))
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)


# Debug: Show thresholded image before contour detection
cv2.imshow("Thresholded Image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find the largest contour (should be the OMR sheet border)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Always use the largest contour for the OMR sheet border
for idx, c in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    print(f"Contour {idx}: area={area}, bbox=({x},{y},{w},{h})")

# Use the largest contour (contours[0])
sheet_contour = contours[0]
peri = cv2.arcLength(sheet_contour, True)
approx = cv2.approxPolyDP(sheet_contour, 0.02 * peri, True)
if len(approx) != 4:
    print(f"Warning: Largest contour does not have 4 corners (has {len(approx)}). Approximating to 4 points.")
    # Use minAreaRect to get 4 corners
    rect = cv2.minAreaRect(sheet_contour)
    box = cv2.boxPoints(rect)
    sheet_cnt = box.astype(int)
else:
    sheet_cnt = approx


# Perspective transform to get a top-down view
def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

rect = order_points(sheet_cnt)
(tl, tr, br, bl) = rect
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(resized_image, M, (maxWidth, maxHeight))

# Debug: Show warped image and print its shape
print(f"Warped image shape: {warped.shape}")
cv2.imshow("Warped OMR Sheet", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Crop the answer area (manually tune these values for your image)
answer_area = warped[80:1620, 60:1140]  # adjust as needed for your image
if answer_area.size == 0:
    print("Error: Cropped answer_area is empty. Please adjust the crop coordinates.")
    exit()
answer_area_gray = cv2.cvtColor(answer_area, cv2.COLOR_BGR2GRAY)
answer_area_blur = cv2.GaussianBlur(answer_area_gray, (5, 5), 0)
answer_area_thresh = cv2.adaptiveThreshold(answer_area_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

# Find all bubbles
contours, _ = cv2.findContours(answer_area_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bubble_contours = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    ar = w / float(h)
    if 18 < w < 40 and 18 < h < 40 and 0.7 < ar < 1.3 and area > 200:
        bubble_contours.append(c)

print(f"Found {len(bubble_contours)} bubble candidates")

# Group and sort bubbles into questions (5 columns, 32 rows, 4 options each)
def sort_bubbles(bubble_contours):
    # Sort by x (columns)
    bubble_contours = sorted(bubble_contours, key=lambda c: cv2.boundingRect(c)[0])
    columns = [[] for _ in range(5)]
    col_width = (answer_area.shape[1] - 60) // 5
    for c in bubble_contours:
        x, y, w, h = cv2.boundingRect(c)
        col_idx = min(4, max(0, (x - 20) // col_width))
        columns[col_idx].append(c)
    # Sort each column by y (top to bottom)
    for i in range(5):
        columns[i] = sorted(columns[i], key=lambda c: cv2.boundingRect(c)[1])
    return columns

columns = sort_bubbles(bubble_contours)

extracted_answers = []
for col_idx, col in enumerate(columns):
    for row in range(32):
        q_bubbles = col[row*4:(row+1)*4]
        if len(q_bubbles) != 4:
            extracted_answers.append('-')
            continue
        filled_pixels = []
        for i, bubble in enumerate(q_bubbles):
            mask = np.zeros(answer_area_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            filled = cv2.countNonZero(cv2.bitwise_and(answer_area_thresh, answer_area_thresh, mask=mask))
            filled_pixels.append(filled)
        max_val = max(filled_pixels)
        second_max = sorted(filled_pixels, reverse=True)[1] if len(filled_pixels) > 1 else 0
        if max_val < 200:  # not filled
            extracted_answers.append('-')
            continue
        most_filled = [(val == max_val and (max_val - second_max) > (0.15 * max_val)) for val in filled_pixels]
        if sum(most_filled) == 1:
            max_index = filled_pixels.index(max_val)
            extracted_answers.append(chr(65 + max_index))
        else:
            extracted_answers.append('-')

print("\nExtracted Answers:")
for i, ans in enumerate(extracted_answers, 1):
    print(f"Q{i}: {ans}")

# --- Score Calculation ---
score = 0
num_questions = min(len(random_answer_key), len(extracted_answers))
for i in range(num_questions):
    if extracted_answers[i] == random_answer_key[i]:
        score += 1
print(f"\nFinal Score: {score} out of {num_questions}")
