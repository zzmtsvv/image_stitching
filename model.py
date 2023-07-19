import cv2
import numpy as np


class ImageStitching:
    def __init__(self,
                 ratio: float = 0.85,
                 min_match: int = 10,
                 window_size: int = 800) -> None:
        self.ratio = ratio
        self.min_match = min_match
        self.window_size = window_size

        self.sift = cv2.SIFT_create()
    
    def registration(self,
                     image1: np.ndarray,
                     image2: np.ndarray):
        
        keypoint1, descriptor1 = self.sift.detectAndCompute(image1, None)
        keypoint2, descriptor2 = self.sift.detectAndCompute(image2, None)

        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(descriptor1, descriptor2, k=2)

        good_points = []
        good_matches = []

        for match1, match2 in raw_matches:
            if match1.distance < self.ratio * match2.distance:
                good_points.append((match1.trainIdx, match1.queryIdx))
                good_matches.append([match1])
        
        image3 = cv2.drawMatchesKnn(image1, keypoint1, image2, keypoint2, good_matches, None, flags=2)
        cv2.imwrite("matching.jpg", image3)

        if len(good_points) > self.min_match:
            image1_keypoint = np.float32(
                [keypoint1[i].pt for (_, i) in good_points]
            )

            image2_keypoint = np.float32(
                [keypoint2[i].pt for (i, _) in good_points]
            )
            H, status = cv2.findHomography(image2_keypoint, image1_keypoint, cv2.RANSAC, 5.0)
        return H
    
    def blending(self,
                 image1: np.ndarray,
                 image2: np.ndarray):
        
        H = self.registration(image1,image2)
        height_img1 = image1.shape[0]
        width_img1 = image1.shape[1]
        width_img2 = image2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(image1,image2,version='left_image')
        panorama1[0:image1.shape[0], 0:image1.shape[1], :] = image1
        panorama1 *= mask1

        mask2 = self.create_mask(image1,image2,version='right_image')
        panorama2 = cv2.warpPerspective(image2, H, (width_panorama, height_panorama)) * mask2
        result = panorama1 + panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result

    def create_mask(self,
                    image1: np.ndarray,
                    image2: np.ndarray,
                    version: str = "left_image"):
        
        height_img1 = image1.shape[0]
        width_img1 = image1.shape[1]
        width_img2 = image2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.window_size / 2)
        barrier = image1.shape[1] - int(self.window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))

        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        
        return cv2.merge([mask, mask, mask])
