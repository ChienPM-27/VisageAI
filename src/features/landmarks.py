import cv2
import mediapipe as mp
import numpy as np

class FaceMeshExtractor:
    def __init__(self, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_landmarks(self, image):
        """
        Extracts 3D landmarks from an image.
        Args:
            image: numpy array (BGR image)
        Returns:
            landmarks: numpy array of shape (N, 3) where N is 468 or 478.
                       x, y are normalized [0, 1], z is depth.
                       Returns None if no face is detected.
        """
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
        ])
        
        # Denormalize to pixel coordinates for x, y
        landmarks_pixel = landmarks.copy()
        landmarks_pixel[:, 0] *= w
        landmarks_pixel[:, 1] *= h
        
        return landmarks_pixel, landmarks, face_landmarks

    def align_face(self, image, landmarks_pixel):
        """
        Aligns the face so that the eyes are horizontal.
        Args:
            image: numpy array (BGR)
            landmarks_pixel: numpy array of shape (N, 3) in pixel coordinates.
        Returns:
            aligned_image: BGR image rotated.
            rotation_matrix: 2x3 transformation matrix.
        """
        # Left eye corner outer: 33, Right eye corner outer: 263
        # Left eye corner inner: 133, Right eye corner inner: 362
        # Center of left eye (average of 33 and 133)
        left_eye = (landmarks_pixel[33][:2] + landmarks_pixel[133][:2]) / 2.0
        # Center of right eye (average of 263 and 362)
        right_eye = (landmarks_pixel[263][:2] + landmarks_pixel[362][:2]) / 2.0
        
        # Calculate angle of rotation
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # We want the eyes to be horizontal, so target angle is 0
        # Center of rotation is midpoint between the eyes
        eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return aligned_image, M
