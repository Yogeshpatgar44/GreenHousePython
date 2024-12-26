from flask import Flask, request
import os

app = Flask(__name__)

# Create a directory to save images if it doesn't exist
UPLOAD_FOLDER = './saved-photos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/saved-photo', methods=['POST'])
def saved_photo():
    try:
        # Check if the incoming request contains image data
        if request.content_type != 'image/jpeg':
            return "Invalid Content-Type. Expecting 'image/jpeg'", 400

        # Read the image data from the request
        image_data = request.data
        if not image_data:
            return "No image data received", 400

        # Save the received image
        image_path = os.path.join(UPLOAD_FOLDER, 'image.jpg')  # Always overwrites with the latest image
        with open(image_path, 'wb') as f:
            f.write(image_data)

        print(f"Image saved at {image_path}")
        return "Image received successfully", 200

    except Exception as e:
        print(f"Error: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
