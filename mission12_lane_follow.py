#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math

class Mission12LaneFollower(object):
    def __init__(self):
        # === 파라미터 ===
        self.image_topic = rospy.get_param("~image_topic", "/limo/color/image_raw")
        self.cmd_topic   = rospy.get_param("~cmd_topic",   "/cmd_vel")

        # ROI: 화면 아래쪽 일부만 사용
        self.roi_y_start_ratio = rospy.get_param("~roi_y_start_ratio", 0.55)

        # 이진화 임계값 (흰색 차선)
        self.binary_threshold = rospy.get_param("~binary_threshold", 180)

        # 조향 관련 파라미터
        self.kp_angular  = rospy.get_param("~kp_angular", 1.5)
        self.max_angular = rospy.get_param("~max_angular", 1.2)

        # 직진 속도 파라미터
        self.base_linear = rospy.get_param("~base_linear", 0.25)
        self.min_linear  = rospy.get_param("~min_linear",  0.10)
        self.max_linear  = rospy.get_param("~max_linear",  0.35)

        # 에러가 클 때 감속
        self.slowdown_gain = rospy.get_param("~slowdown_gain", 1.0)

        # 차선이 안 보일 때 판단 기준
        self.min_lane_area = rospy.get_param("~min_lane_area", 5000)

        # 미션 모드 (1: 지그재그, 2: 급곡선)
        self.mission_mode = rospy.get_param("~mission_mode", 1)
        if self.mission_mode == 2:
            # 급곡선 구간: 더 천천히 + 더 민감하게
            self.base_linear *= 0.8
            self.kp_angular  *= 1.2

        rospy.loginfo("[Mission1&2] image_topic=%s, cmd_topic=%s",
                      self.image_topic, self.cmd_topic)

        # 브리지, 퍼블리셔/서브스크라이버 설정
        self.bridge  = CvBridge()
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image,
                                          self.image_callback, queue_size=1)

        self.show_debug_window = rospy.get_param("~show_debug_window", False)

        # 종료 시 정지시키기 위한 훅
        rospy.on_shutdown(self.on_shutdown)

    def on_shutdown(self):
        """노드 종료(Ctrl+C) 시 로봇 완전히 멈추도록 0 속도 여러 번 발행"""
        stop = Twist()
        for _ in range(5):
            self.cmd_pub.publish(stop)
            rospy.sleep(0.05)
        if self.show_debug_window:
            cv2.destroyAllWindows()
        rospy.loginfo("[Mission1&2] shutdown: stop command sent")

    def image_callback(self, msg: Image):
        """카메라 이미지 콜백 – 여기서 차선을 찾아서 /cmd_vel publish"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn("cv_bridge error: %s", e)
            return

        h, w = frame.shape[:2]

        # ROI 설정 (이미지 아래쪽만 사용)
        y_start = int(h * self.roi_y_start_ratio)
        roi = frame[y_start:h, :]

        # 그레이스케일 + 블러
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 이진화 (흰색 차선)
        _, binary = cv2.threshold(
            gray,
            self.binary_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # 모폴로지(노이즈 제거)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

        # 차선 영역의 무게중심 계산
        M = cv2.moments(binary)
        twist = Twist()

        if M["m00"] < self.min_lane_area:
            # 차선이 거의 안 보임 -> 안전하게 정지
            rospy.logwarn_throttle(1.0, "[Mission1&2] lane not detected, STOP")
            self.cmd_pub.publish(twist)
            if self.show_debug_window:
                cv2.imshow("mission12_binary", binary)
                cv2.waitKey(1)
            return

        # ROI 좌표계에서의 무게중심
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        center_x = w / 2.0
        error_px   = cx - center_x              # px 단위 에러
        error_norm = error_px / center_x        # [-1, 1] 정규화

        # 각속도 (P 제어)
        ang = -self.kp_angular * error_norm
        ang = max(-self.max_angular, min(self.max_angular, ang))

        # 에러가 클수록 감속
        speed_scale = 1.0 - self.slowdown_gain * abs(error_norm)
        speed_scale = max(0.3, speed_scale)

        lin = self.base_linear * speed_scale
        lin = max(self.min_linear, min(self.max_linear, lin))

        twist.linear.x  = lin
        twist.angular.z = ang
        self.cmd_pub.publish(twist)

        # 디버그 출력
        rospy.loginfo_throttle(
            0.5,
            "[Mission1&2] cx=%d err=%.3f lin=%.2f ang=%.2f",
            cx, error_norm, lin, ang
        )

        if self.show_debug_window:
            debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(debug,
                     (int(center_x), 0),
                     (int(center_x), debug.shape[0]-1),
                     (255, 0, 0), 2)
            cv2.imshow("mission12_binary", debug)
            cv2.waitKey(1)

def main():
    rospy.init_node("mission12_lane_follow")
    node = Mission12LaneFollower()
    rospy.spin()

if __name__ == "__main__":
    main()
