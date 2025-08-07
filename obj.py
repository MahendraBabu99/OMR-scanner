import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\chapa mahindra\OneDrive\Pictures\varshithaomr3.jpg")
height, width = image.shape[:2]

# Define regions (approximate percentages based on the image)
roll_number_y1 = int(height * 0.02)  # Top 2% of image
roll_number_y2 = int(height * 0.15)  # Down to 15% of image
roll_number_x1 = int(width * 0.02)   # Left 2% of image
roll_number_x2 = int(width * 0.25)   # Right 25% of image

# Crop roll number region
roll_region = image[roll_number_y1:roll_number_y2, roll_number_x1:roll_number_x2]

# Show roll number region
cv2.imshow("Roll Number Region", roll_region)

# Now process the answer region (everything below roll number section)
answer_y1 = roll_number_y2  # Start from where roll number ends
answer_region = image[answer_y1:, :]  # Take all the way to bottom

# Convert answer region to grayscale
answer_gray = cv2.cvtColor(answer_region, cv2.COLOR_BGR2GRAY)

# Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
answer_gray = clahe.apply(answer_gray)

# Process the answer region
answer_blur = cv2.GaussianBlur(answer_gray, (3, 3), 0)

# Apply thresholding
_, answer_thresh = cv2.threshold(answer_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Show preprocessing steps
cv2.imshow("Answer Region", answer_region)
cv2.imshow("Thresholded Answer Region", answer_thresh)

# Clean up the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
answer_thresh = cv2.morphologyEx(answer_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
answer_thresh = cv2.morphologyEx(answer_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find contours in answer region
answer_contours, _ = cv2.findContours(answer_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy for drawing
answer_debug = answer_region.copy()

# Find and filter bubbles
answer_contours, _ = cv2.findContours(answer_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bubble_contours = []
answer_debug = answer_region.copy()

for c in answer_contours:
    x_, y_, w_, h_ = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    # Calculate circularity
    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Calculate aspect ratio
    aspect_ratio = w_ / h_ if h_ != 0 else 0
    
    # Debug info for each contour
    print(f"Contour - Width: {w_}, Height: {h_}, Area: {area}, Circularity: {circularity:.2f}, Aspect Ratio: {aspect_ratio:.2f}")
    
    # Filter bubbles with stricter circularity and more specific size constraints
    if (15 < w_ < 25 and 15 < h_ < 25 and      # More specific size range for bubbles
        0.8 < aspect_ratio < 1.2 and           # Stricter aspect ratio (more circular)
        circularity > 0.6 and                  # Higher circularity threshold
        area > 200 and area < 500):            # More specific area range for bubbles
        bubble_contours.append(c)
        # Draw detected bubbles in green
        cv2.drawContours(answer_debug, [c], -1, (0, 255, 0), 1)
        # Add bubble center point in red
        center = (x_ + w_//2, y_ + h_//2)
        cv2.circle(answer_debug, center, 1, (0, 0, 255), -1)

# Sort bubbles by position (top to bottom, then left to right)
sorted_bubbles = sorted(bubble_contours, 
                       key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

print(f"Total bubbles detected: {len(bubble_contours)}")

# Show all processing steps
cv2.imshow("Roll Number Region", roll_region)
cv2.imshow("Answer Region", answer_region)
cv2.imshow("Thresholded Answer Region", answer_thresh)
cv2.imshow("Detected Bubbles", answer_debug)

# Process bubbles to extract answers
answers = []
for i in range(0, len(sorted_bubbles), 4):  # Process 4 bubbles at a time
    if i + 4 <= len(sorted_bubbles):
        options = sorted_bubbles[i:i+4]
        # Sort options left to right
        options = sorted(options, key=lambda c: cv2.boundingRect(c)[0])
        
        # Check which bubble is filled
        max_filled = 0
        answer = 'X'  # Default to X if no bubble is clearly filled
        pixel_values = []
        
        # Get pixel values for all options
        for idx, bubble in enumerate(options):
            mask = np.zeros(answer_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            filled = cv2.countNonZero(cv2.bitwise_and(answer_thresh, mask))
            pixel_values.append(filled)
            
            if filled > max_filled and filled > 50:  # Minimum threshold for filled bubble
                max_filled = filled
                answer = chr(65 + idx)  # Convert to A, B, C, D
        
        # Print pixel values and chosen answer for each question
        question_num = i // 4 + 1
        print(f"\nQuestion {question_num}:")
        print(f"A: {pixel_values[0]} pixels")
        print(f"B: {pixel_values[1]} pixels")
        print(f"C: {pixel_values[2]} pixels")
        print(f"D: {pixel_values[3]} pixels")
        print(f"Chosen Answer: {answer} (Pixel value: {max_filled})")
        
        # Draw the answer on the debug image
        for idx, bubble in enumerate(options):
            x, y, w, h = cv2.boundingRect(bubble)
            # Draw pixel values above each bubble
            cv2.putText(answer_debug, str(pixel_values[idx]), 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # Highlight chosen answer in blue
            if chr(65 + idx) == answer:
                cv2.drawContours(answer_debug, [bubble], -1, (255, 0, 0), 2)
        
        answers.append(answer)

print(f"Answers detected: {answers}")
cv2.waitKey(0)
cv2.destroyAllWindows()

