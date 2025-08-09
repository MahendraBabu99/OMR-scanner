import cv2
import numpy as np

def show_image(title, image, wait=True):
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 15)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return cleaned

def remove_top_percentage(image, percent=14):
    height = image.shape[0]
    remove_height = int(height * percent / 100)
    return image[remove_height:, :]

def detect_and_analyze_bubbles(image):
    processed = preprocess_image(image)
    
    # Find all contours
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for only bubbles
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Bubble detection parameters
        if (100 < area < 1000 and 15 < w < 50 and 15 < h < 50 and 0.8 < (w/h) < 1.2):
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            pixels = processed[y:y+h, x:x+w]
            
            filled = cv2.countNonZero(pixels)
            fill_ratio = filled / (w * h)
            
            bubbles.append({
                'position': (x, y),
                'size': (w, h),
                'pixels': pixels,
                'fill_ratio': fill_ratio
            })
    
    # Sort bubbles by position (top-to-bottom, then left-to-right)
    bubbles.sort(key=lambda b: (b['position'][1], b['position'][0]))
    return bubbles

def group_and_analyze_questions(bubbles, options_per_question=4):
    questions = []
    for i in range(0, len(bubbles), options_per_question):
        group = bubbles[i:i+options_per_question]
        if len(group) == options_per_question:
            questions.append(group)
    
    # Analyze each question
    for q_num, question in enumerate(questions[:50], 1):  # First 50 questions
        print(f"\nQuestion {q_num}:")
        for opt_num, bubble in enumerate(question):
            option = chr(97 + opt_num)  # a, b, c, d
            print(f"  Option {option}:")
            print(f"    Position: {bubble['position']}")
            print(f"    Size: {bubble['size']}")
            print(f"    Fill ratio: {bubble['fill_ratio']:.2f}")
            print("    Pixel values (5x5 sample):")
            print(bubble['pixels'][:5, :5])
        
        # Determine marked answer
        marked = max(question, key=lambda b: b['fill_ratio'])
        if marked['fill_ratio'] > 0.4:
            print(f"  Marked answer: {chr(97 + question.index(marked))}")

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Remove top 20% to eliminate header
    cropped_img = remove_top_percentage(img)
    
    # Detect and analyze all bubbles
    bubbles = detect_and_analyze_bubbles(cropped_img)
    print(f"Detected {len(bubbles)} bubbles total")
    
    # Group into questions and analyze
    group_and_analyze_questions(bubbles)
    
    # Visualize results
    result_img = cropped_img.copy()
    for bubble in bubbles:
        x, y = bubble['position']
        cv2.rectangle(result_img, (x, y), (x+bubble['size'][0], y+bubble['size'][1]), (0,255,0), 1)
    show_image("All Detected Bubbles", result_img)

if __name__ == "__main__":
    image_path = r"C:\Users\chapa mahindra\OneDrive\Pictures\eamcet omr2.jpg"
    main(image_path)
