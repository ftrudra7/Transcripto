import logging
import cv2
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)

class SignService:
    def __init__(self):
        self.mp_hands = None
        self.hands_detector = None
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        except (ImportError, AttributeError) as e:
            logger.error(f"mediapipe not available or incompatible: {e}. SignService image features will be limited.")

    async def predict_sign_from_landmarks(self, landmarks: list) -> dict:
        """
        Predicts sign language text from a list of landmarks.
        """
        if not landmarks:
            return {"text": "", "score": 0.0}
            
        logger.info(f"Received {len(landmarks)} landmarks for sign prediction.")
        
        # Mock prediction logic based on landmarks presence
        return {
            "text": "Hello (Mock)",
            "score": 0.85
        }

    async def predict_sign_from_image(self, image_bytes: bytes) -> dict:
        """
        Processes an image, extracts landmarks using MediaPipe, and predicts sign language.
        """
        if not self.hands_detector:
            return {"text": "MediaPipe not available", "score": 0.0, "landmarks_extracted": []}

        try:
            # Decode image bytes to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
                
            # Preprocessing: resize if too large, convert to RGB for MediaPipe
            h, w = img.shape[:2]
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(img_rgb)
            
            extracted_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
                    extracted_landmarks.append(points)
            
            if extracted_landmarks:
                # Mock prediction from extracted landmarks
                return {
                    "text": "Sign Detected (Mock)",
                    "score": 0.90,
                    "landmarks_extracted": extracted_landmarks
                }
            else:
                return {
                    "text": "No hands detected",
                    "score": 0.0,
                    "landmarks_extracted": []
                }
                
        except Exception as e:
            logger.error(f"Error predicting from image: {e}")
            raise

sign_service = SignService()
