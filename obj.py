import cv2
import numpy as np

def show_image(title, image, wait=True):
    cv2.imshow(title, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 15)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return cleaned

def remove_top_percentage(image, percent=20):
    height = image.shape[0]
    remove_height = int(height * percent / 100)
    return image[remove_height:, :]

def analyze_answer_area(image):
    processed = preprocess_image(image)
    
    # Find all contours
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and analyze contours
    contour_data = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 25:  # Skip very small contours
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Get pixel values
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        pixels = processed[y:y+h, x:x+w]
        
        # Calculate fill percentage
        filled = cv2.countNonZero(pixels)
        fill_ratio = filled / (w * h)
        
        contour_data.append({
            'id': i,
            'position': (x, y),
            'size': (w, h),
            'area': area,
            'fill_ratio': fill_ratio,
            'pixels': pixels
        })
    
    return contour_data

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Remove top 20% of the image
    cropped_img = remove_top_percentage(img)
    show_image("Cropped Image (Top 20% Removed)", cropped_img, wait=False)
    
    # Analyze answer area
    contours = analyze_answer_area(cropped_img)
    
    # Print contour data
    print(f"\nDetected {len(contours)} contours in answer area")
    for i, cnt in enumerate(contours[:10]):  # Print first 10 as examples
        print(f"\nContour {i+1}:")
        print(f"Position: {cnt['position']}")
        print(f"Size: {cnt['size']}")
        print(f"Area: {cnt['area']}")
        print(f"Fill ratio: {cnt['fill_ratio']:.2f}")
        print("Pixel values (5x5 sample):")
        print(cnt['pixels'][:5, :5])
    
    # Visualize all contours
    contour_img = cropped_img.copy()
    for cnt in contours:
        x, y = cnt['position']
        w, h = cnt['size']
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0,255,0), 1)
    show_image("All Contours", contour_img)

if __name__ == "__main__":
    image_path = r"C:\Users\chapa mahindra\OneDrive\Pictures\eamcet omr2.jpg"
    main(image_path)
