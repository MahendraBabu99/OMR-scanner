import cv2
import numpy as np

def sort_contours_by_columns(cnts, rows=4, col_tolerance=30):
    """
    Group contours column-wise, then sort top to bottom in each column.
    """
    # Sort contours by horizontal position (left to right)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
    
    # Group into columns based on horizontal proximity
    columns = []
    current_col = []
    last_x = None

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if last_x is None or abs(x - last_x) < col_tolerance:
            current_col.append(c)
        else:
            # Sort current column top to bottom before adding
            current_col = sorted(current_col, key=lambda c: cv2.boundingRect(c)[1])
            columns.append(current_col)
            current_col = [c]
        last_x = x
    
    # Add the last column
    if current_col:
        current_col = sorted(current_col, key=lambda c: cv2.boundingRect(c)[1])
        columns.append(current_col)

    # Organize into questions (groups of 4 rows per column)
    sorted_questions = []
    for col in columns:
        for i in range(0, len(col), rows):
            group = col[i:i + rows]
            if len(group) == rows:
                sorted_questions.append(group)
    
    return sorted_questions

# --- Main processing ---
image = cv2.imread(r"C:\Users\chapa mahindra\Downloads\omr.jpg")
resized = cv2.resize(image, (700, 700))

rois = cv2.selectROIs("Select ROIs", resized, showCrosshair=True)
cv2.destroyAllWindows()

if len(rois) == 0:
    print(" No ROI selected.")
    exit()

print(f" Selected ROIs: {rois}")

question_counter = 1

for roi_idx, (x, y, w, h) in enumerate(rois):
    roi = resized[y:y + h, x:x + w]
    roi = cv2.resize(roi, (800, 800))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []

    for c in contours:
        x_, y_, w_, h_ = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w_ / float(h_)

        if 15 < w_ < 60 and 15 < h_ < 60 and 0.6 < ar < 1.4 and area > 80:
            bubble_contours.append(c)

    print(f"\n ROI {roi_idx + 1}: Found {len(bubble_contours)} bubble candidates")

    if len(bubble_contours) < 4:
        print(" Not enough bubbles found.")
        continue

    # Sort into columns and groups of 4 (A-D options per question)
    questions = sort_contours_by_columns(bubble_contours, rows=4)

    print(f" ROI {roi_idx + 1}: {len(questions)} questions found (column-wise)")

    for q_bubbles in questions:
        if len(q_bubbles) != 4:
            continue  # Skip incomplete groups

        # Bubbles are already sorted top to bottom in column
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
            mark = "" if most_filled else ""
            print(f"    Option {chr(65 + i)}: {val} {mark}")

        question_counter += 1

    # Visualization
    debug_img = roi.copy()
    for i, c in enumerate(bubble_contours):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i+1), (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imshow(f"ROI {roi_idx+1} Bubbles", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
