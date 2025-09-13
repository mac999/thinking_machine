# title: Facial Landmarks Detection with MediaPipe
# author: taewook kang (laputa99999@gmail.com)
# date: 2024-08-16
# install: pip install mediapipe opencv-python matplotlib
#          Requires an image file "man.png"
import cv2
import matplotlib.pyplot as plt
import mediapipe

img_base = cv2.imread("man.png")
img = img_base.copy()

plt.imshow(img[:, :, ::-1])

# Facial landmarks
faceModule = mediapipe.solutions.face_mesh
face_mesh = faceModule.FaceMesh(static_image_mode=True)
results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

landmarks = results.multi_face_landmarks[0]

facial_areas = {
	'Contours': faceModule.FACEMESH_CONTOURS
	, 'Lips': faceModule.FACEMESH_LIPS
	, 'Face_oval': faceModule.FACEMESH_FACE_OVAL
	, 'Left_eye': faceModule.FACEMESH_LEFT_EYE
	, 'Left_eye_brow': faceModule.FACEMESH_LEFT_EYEBROW
	, 'Right_eye': faceModule.FACEMESH_RIGHT_EYE
	, 'Right_eye_brow': faceModule.FACEMESH_RIGHT_EYEBROW
	, 'Tesselation': faceModule.FACEMESH_TESSELATION
}

def plot_landmark(img_base, facial_area_name, facial_area_obj):
	
	print(facial_area_name, ":")
	
	img = img_base.copy()
	
	for source_idx, target_idx in facial_area_obj:
		source = landmarks.landmark[source_idx]
		target = landmarks.landmark[target_idx]

		relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
		relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

		cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 10)
	
	fig = plt.figure(figsize = (15, 15))
	plt.axis('off')
	plt.imshow(img[:, :, ::-1])
	plt.show()

for facial_area in facial_areas.keys():
	facial_area_obj = facial_areas[facial_area]
	plot_landmark(img_base, facial_area, facial_area_obj)

img_lips = img_base.copy()
lips_landmarks = facial_areas['Lips']

for source_idx, target_idx in lips_landmarks:
    source = landmarks.landmark[source_idx]
    target = landmarks.landmark[target_idx]

    relative_source = (int(img_lips.shape[1] * source.x), int(img_lips.shape[0] * source.y))
    relative_target = (int(img_lips.shape[1] * target.x), int(img_lips.shape[0] * target.y))

    cv2.line(img_lips, relative_source, relative_target, (0, 0, 255), thickness=1)

fig = plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(img_lips[:, :, ::-1])
plt.show()

