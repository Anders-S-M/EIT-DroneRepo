# -*- coding: utf-8 -*-
"""
Marker tracker for locating n-fold edges in images using convolution.

@author: Henrik Skov Midtiby
"""
import cv2
import numpy as np
import math
from .MarkerPose import MarkerPose


class MarkerTracker:
    """
    Purpose: Locate a certain marker in an image.
    """

    def __init__(self, order, kernel_size, scale_factor):
        self.kernel_size = kernel_size
        (kernel_real, kernel_imag) = self.generate_symmetry_detector_kernel(order, kernel_size)

        self.order = order
        self.mat_real = np.ascontiguousarray(kernel_real / scale_factor, dtype=np.float32)
        self.mat_imag = np.ascontiguousarray(kernel_imag / scale_factor, dtype=np.float32)

        self.frame_real = None
        self.frame_imag = None
        self.last_marker_location = None
        self.orientation = None
        self.track_marker_with_missing_black_leg = True

        # Create kernel used to remove arm in quality-measure
        kernel_remove_arm_real, kernel_remove_arm_imag = self.generate_symmetry_detector_kernel(1, self.kernel_size)
        self.kernelComplex = np.ascontiguousarray(kernel_real + 1j * kernel_imag, dtype=np.complex64)
        self.KernelRemoveArmComplex = np.ascontiguousarray(kernel_remove_arm_real + 1j * kernel_remove_arm_imag, dtype=np.complex64)

        # Values used in quality-measure
        absolute = np.abs(self.kernelComplex)
        self.threshold = 0.4 * absolute.max()
        self.quality = None
        self.y1 = int(math.floor(float(self.kernel_size) / 2))
        self.y2 = int(math.ceil(float(self.kernel_size) / 2))
        self.x1 = int(math.floor(float(self.kernel_size) / 2))
        self.x2 = int(math.ceil(float(self.kernel_size) / 2))

        # Information about the located marker.
        self.pose = None

    @staticmethod
    def generate_symmetry_detector_kernel(order, kernel_size):
        # type: (int, int) -> (np.ndarray, np.ndarray)
        value_range = np.linspace(-1, 1, kernel_size, dtype=np.float32)
        temp1 = np.meshgrid(value_range, value_range)
        kernel = np.ascontiguousarray(temp1[0] + 1j * temp1[1], dtype=np.complex64)

        magnitude = np.abs(kernel)
        kernel = np.power(kernel, order, dtype=np.complex64)
        kernel = kernel * np.exp(-8 * magnitude**2, dtype=np.complex64)

        return np.real(kernel).astype(np.float32), np.imag(kernel).astype(np.float32)

    def locate_marker(self, frame):
        assert len(frame.shape) == 2, "Input image is not a single-channel image."
        self.frame_real = np.ascontiguousarray(frame.copy(), dtype=np.float32)
        self.frame_imag = np.ascontiguousarray(frame.copy(), dtype=np.float32)

        # Calculate convolution and determine response strength.
        self.frame_real = cv2.filter2D(self.frame_real, cv2.CV_32F, self.mat_real)
        self.frame_imag = cv2.filter2D(self.frame_imag, cv2.CV_32F, self.mat_imag)
        frame_real_squared = cv2.multiply(self.frame_real, self.frame_real, dtype=cv2.CV_32F)
        frame_imag_squared = cv2.multiply(self.frame_imag, self.frame_imag, dtype=cv2.CV_32F)
        self.frame_sum_squared = cv2.add(frame_real_squared, frame_imag_squared, dtype=cv2.CV_32F)

        _, max_val, _, max_loc = cv2.minMaxLoc(self.frame_sum_squared)
        self.last_marker_location = max_loc
        self.determine_marker_orientation(frame)
        self.determine_marker_quality(frame)

        self.pose = MarkerPose(max_loc[0], max_loc[1], self.orientation, self.quality, self.order)
        return self.pose

    def determine_marker_orientation(self, frame):
        xm, ym = self.last_marker_location
        real_value = self.frame_real[ym, xm]
        imag_value = self.frame_imag[ym, xm]
        self.orientation = (math.atan2(-real_value, imag_value) - math.pi / 2) / self.order

        max_value = 0
        max_orientation = 0
        search_distance = self.kernel_size / 3
        for k in range(self.order):
            orient = self.orientation + 2 * k * math.pi / self.order
            xm2 = int(xm + search_distance * math.cos(orient))
            ym2 = int(ym + search_distance * math.sin(orient))
            if 0 <= xm2 < frame.shape[1] and 0 <= ym2 < frame.shape[0]:
                intensity = frame[ym2, xm2]
                if intensity > max_value:
                    max_value = intensity
                    max_orientation = orient

        self.orientation = self.limit_angle_to_range(max_orientation)

    @staticmethod
    def limit_angle_to_range(angle):
        while angle < -math.pi:
            angle += 2 * math.pi
        while angle >= math.pi:
            angle -= 2 * math.pi
        return angle

    def determine_marker_quality(self, frame):
        bright_regions, dark_regions = self.generate_template_for_quality_estimator()

        try:
            frame_img = self.extract_window_around_marker_location(frame)
            bright_mean, bright_std = cv2.meanStdDev(frame_img, mask=bright_regions)
            dark_mean, dark_std = cv2.meanStdDev(frame_img, mask=dark_regions)

            mean_difference = bright_mean - dark_mean
            normalized_mean_difference = mean_difference / (0.5 * bright_std + 0.5 * dark_std)
            self.quality = 1 - 1 / (1 + math.exp(0.75 * (-7 + normalized_mean_difference)))
        except Exception:
            self.quality = 0.0

    def extract_window_around_marker_location(self, frame):
        xm, ym = self.last_marker_location
        frame_tmp = np.ascontiguousarray(frame[ym - self.y1:ym + self.y2, xm - self.x1:xm + self.x2], dtype=np.uint8)
        return frame_tmp

    def generate_template_for_quality_estimator(self):
        phase = np.exp(self.limit_angle_to_range(-self.orientation) * 1j, dtype=np.complex64)
        angle_threshold = math.pi / (2 * self.order)
        t3 = np.angle(self.KernelRemoveArmComplex * phase) < angle_threshold
        t4 = np.angle(self.KernelRemoveArmComplex * phase) > -angle_threshold

        signed_mask = 1 - 2 * (t3 & t4).astype(np.uint8)
        adjusted_kernel = self.kernelComplex * np.power(phase, self.order, dtype=np.complex64)
        if self.track_marker_with_missing_black_leg:
            adjusted_kernel *= signed_mask
        bright_regions = (adjusted_kernel.real < -self.threshold).astype(np.uint8)
        dark_regions = (adjusted_kernel.real > self.threshold).astype(np.uint8)

        return bright_regions, dark_regions
