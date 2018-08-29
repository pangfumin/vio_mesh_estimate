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
 * @file utils.cc
 * @author W. Nicholas Greene
 * @date 2016-12-13 15:28:50 (Tue)
 */

#include "ros_utils.h" // NOLINT

#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>

#include <depth_mesh/utils/image_utils.h>
#include <depth_mesh/utils/visualization.h>


namespace fu = flame::utils;

namespace flame {



void publishDepthMesh(const ros::Publisher& mesh_pub,
                      const std::string& frame_id,
                      double time,
                      const Eigen::Matrix3f& Kinv,
                      const std::vector<cv::Point2f>& vertices,
                      const std::vector<float>& idepths,
                      const std::vector<Eigen::Vector3f>& normals,
                      const std::vector<flame::Triangle>& triangles,
                      const std::vector<bool>& tri_validity,
                      const cv::Mat3b& rgb) {
  pcl_msgs::PolygonMesh::Ptr msg(new pcl_msgs::PolygonMesh());
  msg->header.stamp.fromSec(time);
  msg->header.frame_id = frame_id;

  // Create point cloud to hold vertices.
  pcl::PointCloud<flame::PointNormalUV> cloud;
  cloud.width = vertices.size();
  cloud.height = 1;
  cloud.points.resize(vertices.size());
  cloud.is_dense = false;

  for (int ii = 0; ii < vertices.size(); ++ii) {
    float id = idepths[ii];
    if (!std::isnan(id) && (id > 0.0f)) {
      Eigen::Vector3f uhom(vertices[ii].x, vertices[ii].y, 1.0f);
      uhom /= id;
      Eigen::Vector3f p(Kinv * uhom);
      cloud.points[ii].x = p(0);
      cloud.points[ii].y = p(1);
      cloud.points[ii].z = p(2);

      cloud.points[ii].normal_x = normals[ii](0);
      cloud.points[ii].normal_y = normals[ii](1);
      cloud.points[ii].normal_z = normals[ii](2);

      // OpenGL textures range from 0 to 1.
      cloud.points[ii].u = vertices[ii].x / (rgb.cols - 1);
      cloud.points[ii].v = vertices[ii].y / (rgb.rows - 1);
    } else {
      // Add invalid value to skip this point. Note that the initial value
      // is (0, 0, 0), so you must manually invalidate the point.
      cloud.points[ii].x = std::numeric_limits<float>::quiet_NaN();
      cloud.points[ii].y = std::numeric_limits<float>::quiet_NaN();
      cloud.points[ii].z = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
  }

  pcl::toROSMsg(cloud, msg->cloud);

  // NOTE: Header fields need to be filled in after pcl::toROSMsg() call.
  msg->cloud.header = std_msgs::Header();
  msg->cloud.header.stamp.fromSec(time);
  msg->cloud.header.frame_id = frame_id;

  // Fill in faces.
  msg->polygons.reserve(triangles.size());
  for (int ii = 0; ii < triangles.size(); ++ii) {
    if (tri_validity[ii]) {
      pcl_msgs::Vertices vtx_ii;
      vtx_ii.vertices.resize(3);
      vtx_ii.vertices[0] = triangles[ii][2];
      vtx_ii.vertices[1] = triangles[ii][1];
      vtx_ii.vertices[2] = triangles[ii][0];

      msg->polygons.push_back(vtx_ii);
    }
  }

  if (msg->polygons.size() > 0) {
    mesh_pub.publish(msg);
  }

  return;
}

void publishDepthMap(const image_transport::CameraPublisher& pub,
                     const std::string& frame_id,
                     double time, const Eigen::Matrix3f& K,
                     const cv::Mat1f& depth_est) {
  // Publish depthmap.
  std_msgs::Header header;
  header.stamp.fromSec(time);
  header.frame_id = frame_id;

  sensor_msgs::CameraInfo::Ptr cinfo(new sensor_msgs::CameraInfo);
  cinfo->header = header;
  cinfo->height = depth_est.rows;
  cinfo->width = depth_est.cols;
  cinfo->distortion_model = "plumb_bob";
  cinfo->D = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      cinfo->K[ii*3 + jj] = K(ii, jj);
      cinfo->P[ii*4 + jj] = K(ii, jj);
      cinfo->R[ii*3 + jj] = 0.0;
    }
  }
  cinfo->P[3] = 0.0;
  cinfo->P[7] = 0.0;
  cinfo->P[11] = 0.0;
  cinfo->R[0] = 1.0;
  cinfo->R[4] = 1.0;
  cinfo->R[8] = 1.0;

  cv_bridge::CvImage depth_cvi(header, "32FC1", depth_est);

  pub.publish(depth_cvi.toImageMsg(), cinfo);

  return;
}

void publishPointCloud(const ros::Publisher& pub,
                       const std::string& frame_id,
                       double time, const Eigen::Matrix3f& K,
                       const cv::Mat1f& depth_est,
                       float min_depth, float max_depth) {
  int height = depth_est.rows;
  int width = depth_est.cols;

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width = width;
  cloud.height = height;
  cloud.is_dense = false;
  cloud.points.resize(width * height);

  Eigen::Matrix3f Kinv(K.inverse());
  for (int ii = 0; ii < height; ++ii) {
    for (int jj = 0; jj < width; ++jj) {
      int idx = ii*width + jj;

      float depth = depth_est(ii, jj);

      if (std::isnan(depth) || (depth < min_depth) || (depth > max_depth)) {
        // Add invalid value to skip this point. Note that the initial value
        // is (0, 0, 0), so you must manually invalidate the point.
        cloud.points[idx].x = std::numeric_limits<float>::quiet_NaN();
        cloud.points[idx].y = std::numeric_limits<float>::quiet_NaN();
        cloud.points[idx].z = std::numeric_limits<float>::quiet_NaN();
        continue;
      }

      Eigen::Vector3f xyz(jj * depth, ii * depth, depth);
      xyz = Kinv * xyz;

      cloud.points[idx].x = xyz(0);
      cloud.points[idx].y = xyz(1);
      cloud.points[idx].z = xyz(2);
    }
  }

  sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2());
  pcl::toROSMsg(cloud, *msg);

  msg->header = std_msgs::Header();
  msg->header.stamp.fromSec(time);
  msg->header.frame_id = frame_id;

  pub.publish(msg);

  return;
}

void getDepthConfusionMatrix(const cv::Mat1f& idepths, const cv::Mat1f& depth,
                             cv::Mat1f* idepth_error,  float* total_error,
                             int* true_pos, int* true_neg,
                             int* false_pos, int* false_neg) {
  // Compute confusion matrix with detection being strictly positive idepth.
  *true_pos = 0;
  *true_neg = 0;
  *false_pos = 0;
  *false_neg = 0;

  *total_error = 0.0f;
  *idepth_error = cv::Mat1f(depth.rows, depth.cols,
                            std::numeric_limits<float>::quiet_NaN());
  for (int ii = 0; ii < depth.rows; ++ii) {
    for (int jj = 0; jj < depth.cols; ++jj) {
      if (depth(ii, jj) > 0) {
        if (!std::isnan(idepths(ii, jj))) {
          float idepth_est = idepths(ii, jj);
          float idepth_true = 1.0f / depth(ii, jj);

          float error = fu::fast_abs(idepth_est - idepth_true);
          (*idepth_error)(ii, jj) = error;
          *total_error += error;

          (*true_pos)++;
        } else {
          (*false_neg)++;
        }
      } else if (!std::isnan(idepths(ii, jj))) {
        float idepth_est = idepths(ii, jj);
        float error = fu::fast_abs(idepth_est);
        (*idepth_error)(ii, jj) = error;
        *total_error += error;

        (*false_pos)++;
      } else {
        (*true_neg)++;
      }
    }
  }

  return;
}

}  // namespace flame_ros
