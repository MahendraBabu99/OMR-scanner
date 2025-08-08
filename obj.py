import cv2
import numpy as np

def find_answer_section(image):
    # Convert to grayscale and apply preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find all rectangle-like contours sorted by area (descending)
    rectangles = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            rectangles.append(cnt)
    
    if len(rectangles) < 2:
        return None
    
    # Sort by area and get the second largest (answer section)
    rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)
    answer_section = rectangles[1]  # Skip the first (entire form)
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(answer_section)
    
    # Show the detected answer section
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Detected Answer Section", output)
    
    # Crop to the answer section
    cropped = image[y:y + h, x:x + w]
    return cropped

def get_bubbles(thresh_img):
    # Apply morphological operations to enhance bubbles
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours with RETR_LIST to get all contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # Adjusted bubble detection parameters
        if (150 < area < 3000 and  # Increased area range
            15 < w < 80 and        # Adjusted width range
            15 < h < 80 and        # Adjusted height range
            0.7 < (w/h) < 1.3):    # Aspect ratio for circles
            bubble_contours.append(c)
    
    print(f"[INFO] Total bubbles detected: {len(bubble_contours)}")
    
    # Visualization
    bubble_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bubble_img, bubble_contours, -1, (0, 0, 255), 2)
    cv2.imshow("Detected Bubbles", bubble_img)
    
    return bubble_contours

def group_bubbles(bubble_contours, options_per_row=4):
    # Sort contours left-to-right, then top-to-bottom
    bubble_contours = sorted(bubble_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    
    # Group into questions (each with options_per_row bubbles)
    questions = []
    for i in range(0, len(bubble_contours), options_per_row):
        group = bubble_contours[i:i + options_per_row]
        if len(group) == options_per_row:
            questions.append(group)
    
    return questions

def get_marked_answers(questions, thresh_img, threshold=0.4):
    answers = []
    marked_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    
    for q_index, q_group in enumerate(questions):
        max_fill = 0
        chosen_idx = -1
        
        for idx, c in enumerate(q_group):
            mask = np.zeros(thresh_img.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            bubble_area = cv2.contourArea(c)
            filled_area = cv2.countNonZero(cv2.bitwise_and(thresh_img, thresh_img, mask=mask))
            fill_ratio = filled_area / bubble_area
            
            if fill_ratio > max_fill and fill_ratio > threshold:
                max_fill = fill_ratio
                chosen_idx = idx
        
        if chosen_idx != -1:
            answers.append(chr(65 + chosen_idx))  # A, B, C, D
            cv2.drawContours(marked_img, [q_group[chosen_idx]], -1, (0, 255, 0), 2)
        else:
            answers.append('N')  # No answer marked
    
    cv2.imshow("Marked Answers", marked_img)
    return answers

def process_omr_sheet(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return None
    
    cv2.imshow("Original Image", image)
    
    # Step 1: Find and crop the answer section
    answer_section = find_answer_section(image)
    if answer_section is None:
        print("[ERROR] Could not detect answer section")
        return None
    
    cv2.imshow("Answer Section", answer_section)
    
    # Step 2: Preprocess for bubble detection
    gray = cv2.cvtColor(answer_section, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding - try multiple methods
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Alternative thresholding if needed
    if cv2.countNonZero(thresh) < 100:  # If too dark
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Clean up the image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow("Thresholded Image", thresh)
    
    # Step 3: Detect bubbles
    bubbles = get_bubbles(thresh)
    if not bubbles:
        print("[ERROR] No bubbles detected - trying alternative approach")
        # Try with different parameters
        bubbles = get_bubbles(cv2.bitwise_not(thresh))  # Try inverted image
    
    if not bubbles:
        print("[ERROR] Still no bubbles detected")
        return None
    
    # Step 4: Group bubbles into questions
    questions = group_bubbles(bubbles)
    if not questions:
        print("[ERROR] Could not group bubbles into questions")
        return None
    
    # Step 5: Detect marked answers
    answers = get_marked_answers(questions, thresh)
    
    # Prepare result
    result = {
        'total_questions': len(answers),
        'answered': len([a for a in answers if a != 'N']),
        'answers': answers
    }
    
    print("\n[INFO] Results:")
    print(f"Total Questions: {result['total_questions']}")
    print(f"Answered: {result['answered']}")
    print("Answers:", " ".join(result['answers'][:20]), "...")  # Print first 20 answers
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

if __name__ == '__main__':
    image_path = r"C:\Users\chapa mahindra\OneDrive\Pictures\eamcet omr2.jpg"  # Update with your image path
    result = process_omr_sheet(image_path)
