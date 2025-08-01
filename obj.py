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
image = cv2.imread(r"C:\Users\chapa mahindra\OneDrive\Pictures\finalomr1.jpg")
resized_image = cv2.resize(image, (700, 700))

rois = cv2.selectROIs("Select ROIs", resized_image, showCrosshair=True)
cv2.destroyAllWindows()


if len(rois) == 0:
    print("No ROI selected.")
    exit()

print(f"Selected ROIs: {rois}")
question_counter = 1

# Store extracted answers
extracted_answers = []

for roi_idx, (x, y, w, h) in enumerate(rois):
    roi = resized_image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (1000, 1000))
    # Get the image dimensions
    # height, width = image.shape[:2]

    # # Resize ROI to match image size
    # roi_resized = cv2.resize(roi, (width, height))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    # Morphology to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detect contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_contours = []
    for c in contours:
        x_, y_, w_, h_ = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w_ / float(h_)
        if 15 < w_ < 60 and 15 < h_ < 60 and 0.7 < ar < 1.3 and area > 100:
            bubble_contours.append(c)

    print(f"\nROI {roi_idx + 1}: Found {len(bubble_contours)} bubble candidates")

    if len(bubble_contours) < 4:
        print("Not enough bubbles found.")
        continue

    # Group and sort
    questions = group_questions_column_wise_row_bubbles(bubble_contours)

    print(f"ROI {roi_idx + 1}: {len(questions)} questions found (column-wise, row-bubbles)")

    for q_bubbles in questions:
        if len(q_bubbles) != 4:
            continue

        print(f"\n  Question {question_counter}:")
        filled_pixels = []
        for i, bubble in enumerate(q_bubbles):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            filled = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            filled_pixels.append(filled)

        max_val = max(filled_pixels)
        second_max = sorted(filled_pixels, reverse=True)[1] if len(filled_pixels) > 1 else 0

        for i, val in enumerate(filled_pixels):
            most_filled = val == max_val and (max_val - second_max) > (0.15 * max_val)
            mark = "<-- selected" if most_filled else ""
            print(f"    Option {chr(65 + i)}: {val} {mark}")

        question_counter += 1

        # Print the answer with the highest pixel value
        max_index = filled_pixels.index(max_val)
        print(f"    Answer: Option {chr(65 + max_index)} (pixels: {max_val})\n")

        # Store the extracted answer
        extracted_answers.append(chr(65 + max_index))

        # Print the option with the highest pixel value
        max_index = filled_pixels.index(max_val)
        print(f"    Answer: Option {chr(65 + max_index)} (pixels: {max_val})\n")

    # Debug view
    debug_img = roi.copy()
    for i, c in enumerate(bubble_contours):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i + 1), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow(f"ROI {roi_idx + 1} Bubbles", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Score Calculation ---
score = 0
num_questions = min(len(random_answer_key), len(extracted_answers))
for i in range(num_questions):
    if extracted_answers[i] == random_answer_key[i]:
        score += 1
print(f"\nFinal Score: {score} out of {num_questions}")
