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
 * @file flame_offline_asl.cc
 * @author W. Nicholas Greene
 * @date 2017-07-29 19:32:58 (Sat)
 */

#include <stdio.h>
#include <sys/stat.h>

#include <cstdio>
#include <ctime>
#include <csignal>

#include <memory>
#include <limits>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>

#include <common/kinematics/Transformation.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PolygonMesh.h>

#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>

#include <ros_sensor_streams/asl_rgbd_offline_stream.h>

#include <depth_mesh/flame.h>
#include <depth_mesh/utils/image_utils.h>
#include <depth_mesh/utils/stats_tracker.h>
#include <depth_mesh/utils/load_tracker.h>


#include <../src/ros_utils.h>
#include <../src/depth_estimate.hpp>

namespace bfs = boost::filesystem;
namespace fu = flame::utils;

/**
 * @brief Signal handler to debug crashes.
 */
void crash_handler(int sig) {
  FLAME_ASSERT(false);
  return;
}

using namespace flame;

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "flame");

  // Install signal handlers.
  std::signal(SIGINT, crash_handler);
  std::signal(SIGSEGV, crash_handler);

//  ros::NodeHandle nh("flame");
   ros::NodeHandle pnh("~");

  std::string pose_path;
  getParamOrFail(pnh, "pose_path", &pose_path);

  std::string rgb_path;
  getParamOrFail(pnh, "rgb_path", &rgb_path);

  std::string depth_path;
  getParamOrFail(pnh, "depth_path", &depth_path);

  std::string world_frame_str;
  getParamOrFail(pnh, "world_frame", &world_frame_str);

  ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame world_frame;
  if (world_frame_str == "RDF") {
    world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::RDF;
  } else if (world_frame_str == "FLU") {
    world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::FLU;
  } else if (world_frame_str == "FRD") {
    world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::FRD;
  } else if (world_frame_str == "RFU") {
    world_frame = ros_sensor_streams::ASLRGBDOfflineStream::WorldFrame::RFU;
  } else {
    ROS_ERROR("Unknown world frame!\n");
    return -1;
  }

  // Frame ID of the camera in frame camera_world_frame_id.
  std::string camera_frame_id_;

  // Frame id of the world in camera (Right-Down-Forward) coordinates.
  std::string camera_world_frame_id_;

  getParamOrFail(pnh, "input/camera_frame_id", &camera_frame_id_);
  getParamOrFail(pnh, "input/camera_world_frame_id", &camera_world_frame_id_);

  std::shared_ptr<ros_sensor_streams::ASLRGBDOfflineStream> input
   = std::make_shared<ros_sensor_streams::
          ASLRGBDOfflineStream>(pnh,
                                pose_path,
                                rgb_path,
                                depth_path,
                                "camera",
                                camera_world_frame_id_,
                                camera_frame_id_,
                                world_frame);;

   int num_imgs = 0;
  flame::DepthEstimate node(pnh, input->K(), input->K(),  input->width(), input->height());
  /*==================== Enter main loop ====================*/
  ros::Rate ros_rate(30);

  while (ros::ok() && !input->empty()) {
    uint32_t img_id;
    double time;
    cv::Mat3b rgb;
    cv::Mat1f depth;
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    input->get(&img_id, &time, &rgb, &depth, &q, &t);


    bool isKeyframe = num_imgs % 4;

  // Eat data.
  Eigen::Isometry3d eigen_pose = Eigen::Isometry3d::Identity();
  eigen_pose.linear() = q.toRotationMatrix();
  eigen_pose.translation() = t;
  okvis::kinematics::Transformation pose(eigen_pose.matrix());


  cv::Mat img_gray;
  cv::cvtColor(img_gray, rgb, CV_GRAY2BGR);
  node.processFrame(img_id, okvis::Time(time), pose,
               rgb, rgb, isKeyframe);



    ros::spinOnce();
    ros_rate.sleep();

    num_imgs++;
  }

  if (input->empty()) {
    ROS_INFO("Finished processing.\n");
  } else {
    ROS_ERROR("Unknown error occurred!\n");
  }


  return 0;
}
