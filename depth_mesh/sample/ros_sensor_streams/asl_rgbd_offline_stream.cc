/**
 * This file is part of flame_ros.
 * Copyright (C) 2017 W. Nicholas Greene (wng@csail.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * @file asl_rgbd_offline_stream.cc
 * @author W. Nicholas Greene
 * @date 2017-07-29 19:27:10 (Sat)
 */

#include "./ros_sensor_streams/asl_rgbd_offline_stream.h"

#include <stdio.h>

#include <algorithm>
#include <unordered_set>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/TransformStamped.h>

namespace bfs = boost::filesystem;

namespace du = dataset_utils;
namespace dua = dataset_utils::asl;

namespace ros_sensor_streams {

ASLRGBDOfflineStream::ASLRGBDOfflineStream(ros::NodeHandle& nh,
                                           const std::string& pose_path,
                                           const std::string& rgb0_path,
                                           const std::string& rgb1_path,
                                           const std::string& depth_path,
                                           const std::string& camera_name,
                                           const std::string& camera_world_frame_id,
                                           const std::string& camera_frame_id,
                                           WorldFrame world_frame,
                                           bool publish) :
    nh_(nh),
    inited_(false),
    camera_name_(camera_name),
    camera_world_frame_id_(camera_world_frame_id),
    camera_frame_id_(camera_frame_id),
    world_frame_(world_frame),
    publish_(publish),
    curr_idx_(0),
    pose_data_(pose_path),
    rgb0_data_(rgb0_path),
    rgb1_data_(rgb1_path),
    depth_data_(),
    pose_idxs_(),
    rgb0_idxs_(),
    rgb1_idxs_(),
    depth_idxs_(),
    q_pose_in_body_(),
    t_pose_in_body_(),
    q_cam0_in_body_(),
    t_cam0_in_body_(),
    q_cam1_in_body_(),
    t_cam1_in_body_(),
    width_(),
    height_(),
    K0_(Eigen::Matrix3f::Identity()),
    K1_(Eigen::Matrix3f::Identity()),
    cinfo0_(),
    cinfo1_(),
    intensity_to_depth_factor_(),
    tf_pub_(),
    it_(nh),
    rgb_pub_(),
    depth_pub_() {
  bfs::path depth_path_fs(depth_path);
  if (bfs::exists(depth_path_fs)) {
    // Read in depth data if it exists.
    depth_data_ = std::move(dataset_utils::asl::
                            Dataset<dataset_utils::asl::FileData>(depth_path));
  }

  // Set calibration information.
  width_ = rgb0_data_.metadata()["resolution"][0].as<uint32_t>();
  height_ = rgb0_data_.metadata()["resolution"][1].as<uint32_t>();
  cinfo0_.width = width_;
  cinfo0_.height = height_;
  cinfo0_.distortion_model = "plumb_bob";
  cinfo1_.width = width_;
  cinfo1_.height = height_;
  cinfo1_.distortion_model = "plumb_bob";

  float fu = rgb0_data_.metadata()["intrinsics"][0].as<float>();
  float fv = rgb0_data_.metadata()["intrinsics"][1].as<float>();
  float cu = rgb0_data_.metadata()["intrinsics"][2].as<float>();
  float cv = rgb0_data_.metadata()["intrinsics"][3].as<float>();

  cinfo0_.K = {fu, 0, cu,
              0, fv, cv,
              0, 0, 1};

  K0_(0, 0) = fu;
  K0_(0, 2) = cu;
  K0_(1, 1) = fv;
  K0_(1, 2) = cv;

  cinfo0_.P = {fu, 0, cu, 0,
              0, fv, cv, 0,
              0, 0, 1, 0};

  float k1 = rgb0_data_.metadata()["distortion_coefficients"][0].as<float>();
  float k2 = rgb0_data_.metadata()["distortion_coefficients"][1].as<float>();
  float p1 = rgb0_data_.metadata()["distortion_coefficients"][2].as<float>();
  float p2 = rgb0_data_.metadata()["distortion_coefficients"][3].as<float>();
  float k3 = 0.0f;

  cinfo0_.D = {k1, k2, p1, p2, k3};

  //
  float fu1 = rgb1_data_.metadata()["intrinsics"][0].as<float>();
  float fv1 = rgb1_data_.metadata()["intrinsics"][1].as<float>();
  float cu1 = rgb1_data_.metadata()["intrinsics"][2].as<float>();
  float cv1 = rgb1_data_.metadata()["intrinsics"][3].as<float>();

  cinfo1_.K = {fu1, 0, cu1,
               0, fv1, cv1,
               0, 0, 1};

  K1_(0, 0) = fu1;
  K1_(0, 2) = cu1;
  K1_(1, 1) = fv1;
  K1_(1, 2) = cv1;

  cinfo1_.P = {fu1, 0, cu1, 0,
               0, fv1, cv1, 0,
               0, 0, 1, 0};

  float k11 = rgb1_data_.metadata()["distortion_coefficients"][0].as<float>();
  float k21 = rgb1_data_.metadata()["distortion_coefficients"][1].as<float>();
  float p11 = rgb1_data_.metadata()["distortion_coefficients"][2].as<float>();
  float p21 = rgb1_data_.metadata()["distortion_coefficients"][3].as<float>();
  float k31 = 0.0f;

  cinfo1_.D = {k11, k21, p11, p21, k31};

  if (!depth_data_.path().empty()) {
    intensity_to_depth_factor_ = depth_data_.metadata()["depth_scale_factor"].as<float>();
  }

  if (publish_) {
    rgb_pub_ = it_.advertiseCamera("/" + camera_name_ + "/rgb/image_rect_color", 5);

    if (!depth_data_.path().empty()) {
      depth_pub_ = it_.advertiseCamera("/" + camera_name_ + "/depth_registered/image_rect", 5);
    }
  }

  associateData();

  // Extract transform of pose sensor in body frame.
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_pose_in_body;
  du::readMatrix(pose_data_.metadata(), "T_BS", 4, 4, T_pose_in_body.data());
  q_pose_in_body_ = Eigen::Quaterniond(T_pose_in_body.block<3, 3>(0, 0));
  t_pose_in_body_ = T_pose_in_body.block<3, 1>(0, 3);

  // Extract transform of camera in body frame.
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_cam0_in_body;
  du::readMatrix(rgb0_data_.metadata(), "T_BS", 4, 4, T_cam0_in_body.data());
  q_cam0_in_body_ = Eigen::Quaterniond(T_cam0_in_body.block<3, 3>(0, 0));
  t_cam0_in_body_ = T_cam0_in_body.block<3, 1>(0, 3);

  // Extract transform of camera in body frame.
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_cam1_in_body;
  du::readMatrix(rgb1_data_.metadata(), "T_BS", 4, 4, T_cam1_in_body.data());
  q_cam1_in_body_ = Eigen::Quaterniond(T_cam1_in_body.block<3, 3>(0, 0));
  t_cam1_in_body_ = T_cam1_in_body.block<3, 1>(0, 3);

  return;
}

void ASLRGBDOfflineStream::associateData() {
  auto diff = [](const dua::FileData& x, const dua::PoseData& y) {
    double tx = static_cast<double>(x.timestamp) * 1e-9;
    double ty = static_cast<double>(y.timestamp) * 1e-9;
    return std::fabs(tx - ty);
  };

  // Match rgb and depth to pose separately.
  std::vector<std::size_t> pose_rgb_idxs;
  std::vector<std::size_t> rgb0_pose_idxs;
  du::associate(rgb0_data_.data(), pose_data_.data(), &rgb0_pose_idxs,
                &pose_rgb_idxs, diff);

  std::vector<std::size_t> pose_depth_idxs;
  std::vector<std::size_t> depth_pose_idxs;
  if (!depth_data_.path().empty()) {
    du::associate(depth_data_.data(), pose_data_.data(), &depth_pose_idxs,
                  &pose_depth_idxs, diff);
  } else {
    // No depth data. Just copy rgb data.
    pose_depth_idxs = pose_rgb_idxs;
    depth_pose_idxs = rgb0_pose_idxs;
  }

  // Now take the intersection of pose_rgb_idxs and pose_depth_idxs to get the
  // indices that match both.
  pose_idxs_.clear();
  std::set_intersection(pose_rgb_idxs.begin(), pose_rgb_idxs.end(),
                        pose_depth_idxs.begin(), pose_depth_idxs.end(),
                        std::back_inserter(pose_idxs_));

  // Now get corresponding rgb and depth idxs.
  std::unordered_set<std::size_t> pose_idxs_set(pose_idxs_.begin(), pose_idxs_.end());

  rgb0_idxs_.clear();
  rgb1_idxs_.clear();
  for (int ii = 0; ii < pose_rgb_idxs.size(); ++ii) {
    if (pose_idxs_set.count(pose_rgb_idxs[ii]) > 0) {
      rgb0_idxs_.push_back(rgb0_pose_idxs[ii]);
      rgb1_idxs_.push_back(rgb0_pose_idxs[ii]);
    }
  }

  depth_idxs_.clear();
  if (!depth_data_.path().empty()) {
    for (int ii = 0; ii < pose_depth_idxs.size(); ++ii) {
      if (pose_idxs_set.count(pose_depth_idxs[ii]) > 0) {
        depth_idxs_.push_back(depth_pose_idxs[ii]);
      }
    }
  }

  return;
}

void ASLRGBDOfflineStream::get(uint32_t* id, double* time,
                               cv::Mat3b* rgb0, cv::Mat3b* rgb1,
                               cv::Mat1f* depth,
                               Eigen::Quaterniond* quat0,
                               Eigen::Vector3d* trans0,
                               Eigen::Quaterniond* quat1,
                                Eigen::Vector3d* trans1) {
  // Make sure we actually have data to read in.
  if (empty()) {
    ROS_ERROR("No more data!\n");
    return;
  }

  *id = curr_idx_;
  *time = static_cast<double>(rgb0_data_[rgb0_idxs_[curr_idx_]].timestamp) * 1e-9;

  // Load raw pose, which is the pose of the pose sensor wrt a given world
  // frame.
  Eigen::Quaterniond q_pose_in_world(pose_data_[pose_idxs_[curr_idx_]].quat);
  Eigen::Vector3d t_pose_in_world(pose_data_[pose_idxs_[curr_idx_]].trans);
  q_pose_in_world.normalize();

  // Get pose of camera wrt to world frame.
  Eigen::Quaterniond q_body_in_pose(q_pose_in_body_.inverse());
  Eigen::Vector3d t_body_in_pose(-(q_pose_in_body_.inverse() * t_pose_in_body_));

  Eigen::Quaterniond q_body_in_world(q_pose_in_world * q_body_in_pose);
  Eigen::Vector3d t_body_in_world(q_pose_in_world * t_body_in_pose + t_pose_in_world);

  Eigen::Quaterniond q_cam0_in_world = q_body_in_world * q_cam0_in_body_;
  Eigen::Vector3d t_cam0_in_world = q_body_in_world * t_cam0_in_body_ + t_body_in_world;

Eigen::Quaterniond q_cam1_in_world = q_body_in_world * q_cam1_in_body_;
Eigen::Vector3d t_cam1_in_world = q_body_in_world * t_cam1_in_body_ + t_body_in_world;


// Convert poses to optical coordinates.
  switch (world_frame_) {
    case RDF: {
      *quat0 = q_cam0_in_world;
      *trans0 = t_cam0_in_world;
      break;
    }
    case FLU: {
      // Local RDF frame in global FLU frame.
      Eigen::Quaterniond q_flu_to_rdf(-0.5, -0.5, 0.5, -0.5);
      *quat0 = q_flu_to_rdf * q_cam0_in_world;
      *trans0 = q_flu_to_rdf * t_cam0_in_world;
      break;
    }
    case FRD: {
      // Local RDF frame in global FRD frame.
      Eigen::Matrix3d R_frd_to_rdf;
      R_frd_to_rdf << 0.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          1.0, 0.0, 0.0;

      Eigen::Quaterniond q_frd_to_rdf(R_frd_to_rdf);
      *quat0 = q_frd_to_rdf * q_cam0_in_world;
      *trans0 = q_frd_to_rdf * t_cam0_in_world;
      break;
    }
    case RFU: {
      // Local RDF frame in global RFU frame.
      Eigen::Matrix3d R_rfu_to_rdf;
      R_rfu_to_rdf << 1.0, 0.0, 0.0,
          0.0, 0.0, -1.0,
          0.0, 1.0, 0.0;

      Eigen::Quaterniond q_rfu_to_rdf(R_rfu_to_rdf);
      *quat0 = q_rfu_to_rdf * q_cam0_in_world;
      *trans0 = q_rfu_to_rdf * t_cam0_in_world;

    *quat1 = q_rfu_to_rdf * q_cam1_in_world;
    *trans1 = q_rfu_to_rdf * t_cam1_in_world;
      break;
    }
    default:
      ROS_ERROR("Unknown input frame specified!\n");
      return;
  }

  // Load RGB.
  bfs::path rgb0_path = bfs::path(rgb0_data_.path()) / bfs::path("data") /
      bfs::path(rgb0_data_[rgb0_idxs_[curr_idx_]].filename);
  cv::Mat3b rgb0_raw = cv::imread(rgb0_path.string(), cv::IMREAD_COLOR);



  // Undistort image.
  cv::Mat Kcv0;
  cv::Mat Kcv1;
  eigen2cv(K0_, Kcv0);
  cv::undistort(rgb0_raw, *rgb0, Kcv0, cinfo0_.D);


  bfs::path rgb1_path = bfs::path(rgb1_data_.path()) / bfs::path("data") /
                        bfs::path(rgb1_data_[rgb1_idxs_[curr_idx_]].filename);
  //std::cout<< "rgb1_path: " << rgb1_path.string() << std::endl;
  cv::Mat3b rgb1_raw = cv::imread(rgb1_path.string(), cv::IMREAD_COLOR);
  eigen2cv(K1_, Kcv1);
  cv::undistort(rgb1_raw, *rgb1, Kcv1, cinfo1_.D);

  if (!depth_data_.path().empty()) {
    // Have depth data.
    bfs::path depth_path = bfs::path(depth_data_.path()) / bfs::path("data") /
        bfs::path(depth_data_[depth_idxs_[curr_idx_]].filename);
    cv::Mat_<uint16_t> depth_raw =
        cv::imread(depth_path.string(), cv::IMREAD_ANYDEPTH);

    // Assume depthmaps are undistorted!
    cv::Mat depth_undistorted;
    depth_undistorted = depth_raw.clone();

    // Scale depth information.
    depth->create(height_, width_);
    float* depth_ptr = reinterpret_cast<float*>(depth->data);
    uint16_t* depth_raw_ptr = reinterpret_cast<uint16_t*>(depth_undistorted.data);
    for (uint32_t ii = 0; ii < height_ * width_; ++ii) {
      depth_ptr[ii] = static_cast<float>(depth_raw_ptr[ii]) / intensity_to_depth_factor_;
    }
  }

  if (publish_) {
    // Publish pose over tf.
    geometry_msgs::TransformStamped tf;
    tf.header.stamp.fromSec(*time);
    tf.header.frame_id = camera_world_frame_id_;
    tf.child_frame_id = camera_frame_id_;
    tf.transform.rotation.w = quat0->w();
    tf.transform.rotation.x = quat0->x();
    tf.transform.rotation.y = quat0->y();
    tf.transform.rotation.z = quat0->z();

    tf.transform.translation.x = (*trans0)(0);
    tf.transform.translation.y = (*trans0)(1);
    tf.transform.translation.z = (*trans0)(2);
    tf_pub_.sendTransform(tf);

    // Publish messages over ROS.
    std_msgs::Header header;
    header.stamp.fromSec(*time);
    header.frame_id = camera_frame_id_;

    sensor_msgs::CameraInfo::Ptr cinfo_msg(new sensor_msgs::CameraInfo);
    *cinfo_msg = cinfo0_;
    cinfo_msg->header = header;

    cv_bridge::CvImage rgb_cvi(header, "bgr8", *rgb0);
    rgb_pub_.publish(rgb_cvi.toImageMsg(), cinfo_msg);

    if (!depth_data_.path().empty()) {
      cv_bridge::CvImage depth_cvi(header, "32FC1", *depth);
      depth_pub_.publish(depth_cvi.toImageMsg(), cinfo_msg);
    }
  }

  // Increment counter.
  curr_idx_++;

  return;
}

}  // namespace ros_sensor_streams
