import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
import torch

class FallDetection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = keypointrcnn_resnet50_fpn(pretrained=True, weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.target_width = 500
        self.target_height = 500
        self.transform = T.Compose([T.ToTensor(), T.Resize((500, 500), antialias=True)])
        self.activity_label = {"text": "OK", "color": (20, 200, 30)}
        self.fall_score_label = {"text": "Fall score: ", "fall_score": 0, "color": (20, 200, 30)}
        self.fall_detected = False
        self.fall_score = 0
        self.threshold = 0.32
        self.frame_buffer = {"keypoints": [], "boxes": []}
        self.keypoint_mask = np.array([i for i in range(17)])  # 17 keypoints

    def map_value(self, x, from_min, from_max, to_min, to_max):
        return (x - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

    def calculate_theta(self, a, b):
        if torch.all(a == b):
            return 0
        return torch.arcsin((a[0] - b[0]) / ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) / (2 * np.pi)

    def exponential_weighted_average(self, a, alpha):
        result = []
        temp = 0
        for t, item in enumerate(a):
            temp = (alpha * item + (1 - alpha) * temp) / (1 - alpha ** (t + 1))
            result.append(temp)
        return result

    def calculate_difference(self, x):
        return self.exponential_weighted_average(self.exponential_weighted_average(np.diff(x), 0.1), 0.1)

    def detect(self, video_path=0):
        kinematic_features = []
        height_width_ratio = []
        new_skeleton = []

        cap = cv2.VideoCapture(video_path)

        stat_plot = plt.figure(figsize=(2, 2))

        buffer_size = 9
        frame_count = 0

        connect_skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8), (5, 6), (11, 12),
            (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            height, width, channel = frame.shape
            frame_count += 1

            image_tf = self.transform(frame)
            output = self.model([image_tf.to(self.device)])
            
            if output[0]['keypoints'] == []:
                continue

            stat_plot.canvas.draw()

            if len(output) > 0 and 'keypoints' in output[0] and len(output[0]['keypoints']) > 0:
                keypoints = output[0]['keypoints'][0][self.keypoint_mask][..., :-1]
                boxes = output[0]["boxes"][0][:]
                self.frame_buffer['keypoints'].append(keypoints)
                self.frame_buffer["boxes"].append(boxes)
            else:
                continue
            
            old_skeleton = new_skeleton
            new_skeleton = self.frame_buffer['keypoints'][-1]
            self.frame_buffer["keypoints"][-1] = self.frame_buffer["keypoints"][-1].detach().cpu().numpy()

            image_tf = image_tf.detach().cpu().numpy()
            height_width_ratio.append(abs((self.frame_buffer["boxes"][-1][0] - self.frame_buffer["boxes"][-1][2]).item()/(self.frame_buffer["boxes"][-1][1] - self.frame_buffer["boxes"][-1][3]).item()))

            if new_skeleton != [] and old_skeleton != []:
                res = 0
                for i in range(17):
                    for j in range(i + 1, 17):
                        res += abs(self.calculate_theta(old_skeleton[i], old_skeleton[j]) - self.calculate_theta(new_skeleton[i], new_skeleton[j]))

                kinematic_features.append(res.item())   

            if frame_count > buffer_size:
                del self.frame_buffer["keypoints"][0]
                del self.frame_buffer["boxes"][0]

                if frame_count > 50:
                    del kinematic_features[0]
                    del height_width_ratio[0]
            
            plt.clf()
            plt.xlim(frame_count - 20, frame_count + 15)
            plt.ylim(0, 9)

            kinematic_ewa = self.exponential_weighted_average(kinematic_features, 0.1)
            plt.plot(range(frame_count - len(kinematic_features) + buffer_size, frame_count + buffer_size), kinematic_ewa, color=(0.9, 0.2, 0.2))

            # Check if fall is detected, and set the flag
            self.fall_score = sum(self.calculate_difference(kinematic_ewa[-buffer_size:]))
            self.fall_score_label['fall_score'] = self.fall_score * 100
            if not self.fall_detected:
                if self.fall_score > self.threshold:
                    self.activity_label = {'text': "FALL DETECTED", 'color': (20, 30, 200)}
                    self.fall_score_label = {'text': "Fall Score: ", 'fall_score': self.fall_score * 100, 'color': (20, 30, 200)}
                    self.fall_detected = True

            ratio_ewa = self.exponential_weighted_average(height_width_ratio, 0.1)      
            plt.plot(range(frame_count - len(kinematic_features) - 1 + buffer_size, frame_count + buffer_size), ratio_ewa, color=(0.1, 0.1, 0.9))

            plt.legend(("Cost", "HW_Ratio"), loc="upper right")

            for point in self.frame_buffer["keypoints"][-1]:
                point[0] = self.map_value(point[0], 0, self.target_width, 0, width)
                point[1] = self.map_value(point[1], 0, self.target_height, 0, height)
                point = np.array(point, dtype='int')
                frame = cv2.circle(frame, center=point, radius=3, color=(191, 13, 67), thickness=3)

            for connection in connect_skeleton:
                point1 = self.frame_buffer["keypoints"][-1][connection[0]]
                point2 = self.frame_buffer["keypoints"][-1][connection[1]]
                point1 = np.array((int(point1[0]), int(point1[1])))
                point2 = np.array((int(point2[0]), int(point2[1])))
                frame = cv2.line(frame, tuple(point1), tuple(point2), (0, 0, 0), 2)

            img = np.frombuffer(stat_plot.canvas.tostring_rgb(), dtype=np.uint8, count=-1, offset=0)
            img = img.reshape(stat_plot.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            frame[0:img.shape[0], width - img.shape[1]:width] = img
            cv2.putText(frame, self.activity_label["text"], org=(0, 20), fontFace=2, fontScale=1.0, color=self.activity_label["color"], thickness=2)
            cv2.putText(frame, self.fall_score_label['text'] + str(round(self.fall_score_label['fall_score'], 2)), org=(0, 50),
                        fontFace=2, fontScale=1.0, color=self.fall_score_label['color'], thickness=2)
            cv2.imshow("Output", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return self.fall_detected, self.fall_score

    def __call__(self, skeleton_cache, video_path):
        if skeleton_cache:
            self.frame_buffer['keypoints'] = skeleton_cache.get('keypoints', [])
            self.frame_buffer['boxes'] = skeleton_cache.get('boxes', [])

        fall_detected, fall_score = self.detect(video_path=video_path)

        skeleton_cache['keypoints'] = self.frame_buffer['keypoints']
        skeleton_cache['boxes'] = self.frame_buffer['boxes']

        return fall_detected, fall_score
    
fall_detection_system = FallDetection()
fall_detection_system(fall_detection_system.frame_buffer, 'data/video_3.mp4')