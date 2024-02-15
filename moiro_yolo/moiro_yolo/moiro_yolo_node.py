import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from std_srvs.srv import SetBool

class YOLOv8ROS2Node(Node):
    def __init__(self):
        super().__init__('yolov8_ros2_node')
        # parameter setting -> launch file parameter 값을 사용하기 위해
        self.declare_parameter('model', 'yolov8n.pt')
        model = self.get_parameter('model').get_parameter_value().string_value

        self.declare_parameter('enable', True)
        self.enable = self.get_parameter('enable').get_parameter_value().bool_value

        self.declare_parameter('device', 'cuda:0')
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.cv_bridge = CvBridge()
        self.yolo = YOLO(model) # yolov8n.pt 모델 사용
        self.yolo.fuse() # fuse() 함수를 사용하여 모델을 최적화하고, 추론 속도를 높일 수 있음

        # publisher
        self.publisher = self.create_publisher(
            Image,
            'yolov_image',
            10
        )

        # subscriber
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.img_callback,
            10
        )

        # service
        self.service = self.create_service(
            SetBool,
            'enable',
            self.enable_callback
        )

    def enable_callback(self, req: SetBool.Request, res: SetBool.Response)-> SetBool.Response:
        self.enable = req.data
        res.success = True
        res.message = "YOLOv8 is enabled"
        return res

    def img_callback(self, msg) -> None:
        if self.enable:
            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=0.25,
                device=self.device
            )
            results: Results = results[0].cpu()
        # publish
        detecion_msg = self.cv_bridge.cv2_to_imgmsg(results.plot(), 'bgr8')
        self.publisher.publish(detecion_msg)

def main(args=None):
    rclpy.init(args=args)
    yolov8_ros2_node = YOLOv8ROS2Node()
    rclpy.spin(yolov8_ros2_node)
    yolov8_ros2_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
