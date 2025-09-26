import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16

print("start")

# -----------------------------
# Load the VGG16 model (ImageNet weights)
# -----------------------------
model = VGG16(weights='imagenet', include_top=False,
              input_shape=(224, 224, 3), pooling='avg')

# -----------------------------
# Helper: preprocess image for VGG16
# -----------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)  # same normalization VGG expects
    return img

# -----------------------------
# Compute embedding for one face
# -----------------------------
def get_embedding(img_array):
    embedding = model.predict(np.expand_dims(img_array, axis=0))
    return embedding.flatten()

# -----------------------------
# Known users (precompute embeddings once)
# -----------------------------
known_users_raw = {
    'user1': [preprocess_image('user1_1.jpg'),
              preprocess_image('user1_2.jpg')]
    # Add more users here...
}

known_users = {}
for user, face_imgs in known_users_raw.items():
    embeddings = [get_embedding(img) for img in face_imgs]
    known_users[user] = embeddings

print("Known embeddings computed.")

# -----------------------------
# Camera setup (Jetson CSI pipeline)
# -----------------------------
camSet = ('nvarguscamerasrc ! video/x-raw(memory:NVMM), '
          'width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)30/1 ! '
          'nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! '
          'videoconvert ! video/x-raw, format=(string)BGR ! appsink')

cap = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face).resize((224, 224))
        face = np.array(face)
        face = preprocess_input(face)

        # Compute embedding for detected face
        embedding_face = get_embedding(face)

        # Compare with known embeddings
        similarities = {}
        for user, embeddings in known_users.items():
            scores = []
            for emb in embeddings:
                sim = cosine_similarity(
                    embedding_face.reshape(1, -1),
                    emb.reshape(1, -1)
                )[0][0]
                scores.append(sim)
            similarities[user] = max(scores)  # take best match for that user

        # Decide identity
        identified_user = max(similarities, key=similarities.get)
        confidence = similarities[identified_user]

        # Threshold for "unknown" (tune this)
        if confidence < 0.5:
            identified_user = "Unknown"

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{identified_user} ({confidence:.2f})',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
