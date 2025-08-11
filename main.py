import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the CSV file
csv_path = "color_names.csv"
colors_df = pd.read_csv(csv_path)

# Extract RGB values and names
X = colors_df[['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']].values
y = colors_df['Name'].values

# Train a KNN model for color prediction
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X, y)

# Read the image
img_path = "image2.jpeg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image file not found at {img_path}")

display_img = img.copy()
clicked_info = None

# Function to convert RGB to Hex
def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]).upper()

# Mouse callback function
def get_color_details(event, x, y, flags, param):
    global clicked_info, display_img

    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= img.shape[0] or x >= img.shape[1]:
            return
        
        b, g, r = map(int, img[y, x])  # Ensure integers

        # Predict nearest color name
        color_name_pred = knn_model.predict([[r, g, b]])[0]

        # Find closest matches in dataset
        distances = np.linalg.norm(X - [r, g, b], axis=1)
        closest_indices = np.argsort(distances)[:3]
        examples = ', '.join(colors_df.iloc[closest_indices]['Name'].values)

        # Check if exact RGB match exists
        exact_match = colors_df[
            (colors_df['Red (8 bit)'] == r) &
            (colors_df['Green (8 bit)'] == g) &
            (colors_df['Blue (8 bit)'] == b)
        ]
        if not exact_match.empty:
            hex_code = exact_match.iloc[0]['Hex (24 bit)']
        else:
            try:
                hex_code = rgb_to_hex((r, g, b))
            except ValueError:
                hex_code = f"#{r:02x}{g:02x}{b:02x}"

        clicked_info = {
            "color_name": color_name_pred,
            "rgb": (r, g, b),
            "hex": hex_code,
            "examples": examples
        }

        display_img = img.copy()

        # Position the box so it doesn't go outside image
        box_x1 = min(x + 10, img.shape[1] - 200)
        box_y1 = min(y + 10, img.shape[0] - 80)
        box_x2 = box_x1 + 180
        box_y2 = box_y1 + 70

        # Draw colored rectangle background
        cv2.rectangle(display_img, (box_x1, box_y1), (box_x2, box_y2), (b, g, r), -1)

        # Add text inside the box
        text_x = box_x1 + 5
        text_y = box_y1 + 20
        cv2.putText(display_img, f"{color_name_pred}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(display_img, f"RGB: {r},{g},{b}", (text_x, text_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(display_img, f"Hex: {hex_code}", (text_x, text_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# Create window and set callback
cv2.namedWindow('Color Picker')
cv2.setMouseCallback('Color Picker', get_color_details)

print("Click on any part of the image to get color details. Press 'Esc' to exit.")

while True:
    cv2.imshow('Color Picker', display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
