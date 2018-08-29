#include "depth_estimate.hpp"

DepthEstimate::DepthEstimate() :
        stats_(),
        load_(getpid()),
        nh_("flame"),
        pnh_("~"),
        num_imgs_(0),
        camera_world_frame_id_(),
        camera_frame_id_(),
        output_dir_(),
        pass_in_truth_(false),
        input_(nullptr),
        Kinv_(),
        max_angular_rate_(0.0f),
        prev_time_(0),
        prev_pose_(),
        params_(),
        sensor_(nullptr),
        it_(nh_),
        depth_pub_() {
    /*==================== Data Params ====================*/
    std::string pose_path;
    getParamOrFail(pnh_, "pose_path", &pose_path);

    std::string rgb_path;
    getParamOrFail(pnh_, "rgb_path", &rgb_path);

    std::string depth_path;
    getParamOrFail(pnh_, "depth_path", &depth_path);

    std::string world_frame_str;
    getParamOrFail(pnh_, "world_frame", &world_frame_str);

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
        return;
    }

    /*==================== Input Params ====================*/
    getParamOrFail(pnh_, "input/camera_frame_id", &camera_frame_id_);
    getParamOrFail(pnh_, "input/camera_world_frame_id", &camera_world_frame_id_);
    getParamOrFail(pnh_, "input/subsample_factor", &subsample_factor_);
    getParamOrFail(pnh_, "input/poseframe_subsample_factor",
                   &poseframe_subsample_factor_);

    getParamOrFail(pnh_, "input/resize_factor", &resize_factor_);

    getParamOrFail(pnh_, "input/rate", &rate_);

    /*==================== Output Params ====================*/
    getParamOrFail(pnh_, "output/quiet", &params_.debug_quiet);
    getParamOrFail(pnh_, "output/mesh", &publish_mesh_);
    getParamOrFail(pnh_, "output/idepthmap", &publish_idepthmap_);
    getParamOrFail(pnh_, "output/depthmap", &publish_depthmap_);
    getParamOrFail(pnh_, "output/cloud", &publish_cloud_);
    getParamOrFail(pnh_, "output/features", &publish_features_);
    getParamOrFail(pnh_, "output/stats", &publish_stats_);
    getParamOrFail(pnh_, "output/load_integration_factor",
                   &load_integration_factor_);
    getParamOrFail(pnh_, "output/scene_color_scale", &params_.scene_color_scale);
    getParamOrFail(pnh_, "output/filter_oblique_triangles",
                   &params_.do_oblique_triangle_filter);

    double oblique_normal_thresh;
    getParamOrFail(pnh_, "output/oblique_normal_thresh", &oblique_normal_thresh);
    params_.oblique_normal_thresh = oblique_normal_thresh;

    getParamOrFail(pnh_, "output/oblique_idepth_diff_factor",
                   &params_.oblique_idepth_diff_factor);
    getParamOrFail(pnh_, "output/oblique_idepth_diff_abs",
                   &params_.oblique_idepth_diff_abs);

    getParamOrFail(pnh_, "output/filter_long_edges",
                   &params_.do_edge_length_filter);

    double edge_length_thresh;
    getParamOrFail(pnh_, "output/edge_length_thresh", &edge_length_thresh);
    params_.edge_length_thresh = edge_length_thresh;

    getParamOrFail(pnh_, "output/filter_triangles_by_idepth",
                   &params_.do_idepth_triangle_filter);

    double min_triangle_idepth;
    getParamOrFail(pnh_, "output/min_triangle_idepth", &min_triangle_idepth);
    params_.min_triangle_idepth = min_triangle_idepth;

    getParamOrFail(pnh_, "output/max_angular_rate", &max_angular_rate_);

    /*==================== Debug Params ====================*/
    getParamOrFail(pnh_, "debug/wireframe", &params_.debug_draw_wireframe);
    getParamOrFail(pnh_, "debug/features", &params_.debug_draw_features);
    getParamOrFail(pnh_, "debug/detections", &params_.debug_draw_detections);
    getParamOrFail(pnh_, "debug/matches", &params_.debug_draw_matches);
    getParamOrFail(pnh_, "debug/normals", &params_.debug_draw_normals);
    getParamOrFail(pnh_, "debug/idepthmap", &params_.debug_draw_idepthmap);
    getParamOrFail(pnh_, "debug/text_overlay", &params_.debug_draw_text_overlay);
    getParamOrFail(pnh_, "debug/flip_images", &params_.debug_flip_images);

    /*==================== Threading Params ====================*/
    getParamOrFail(pnh_, "threading/openmp/num_threads", &params_.omp_num_threads);
    getParamOrFail(pnh_, "threading/openmp/chunk_size", &params_.omp_chunk_size);

    /*==================== Features Params ====================*/
    getParamOrFail(pnh_, "features/do_letterbox", &params_.do_letterbox);
    getParamOrFail(pnh_, "features/detection/min_grad_mag", &params_.min_grad_mag);
    params_.fparams.min_grad_mag = params_.min_grad_mag;

    double min_error;
    getParamOrFail(pnh_, "features/detection/min_error", &min_error);
    params_.min_error = min_error;

    getParamOrFail(pnh_, "features/detection/win_size", &params_.detection_win_size);

    int win_size;
    getParamOrFail(pnh_, "features/tracking/win_size", &win_size);
    params_.zparams.win_size = win_size;
    params_.fparams.win_size = win_size;

    getParamOrFail(pnh_, "features/tracking/max_dropouts", &params_.max_dropouts);

    double epipolar_line_var;
    getParamOrFail(pnh_, "features/tracking/epipolar_line_var",
                   &epipolar_line_var);
    params_.zparams.epipolar_line_var = epipolar_line_var;

    /*==================== Regularizer Params ====================*/
    getParamOrFail(pnh_, "regularization/do_nltgv2", &params_.do_nltgv2);
    getParamOrFail(pnh_, "regularization/nltgv2/adaptive_data_weights",
                   &params_.adaptive_data_weights);
    getParamOrFail(pnh_, "regularization/nltgv2/rescale_data", &params_.rescale_data);
    getParamOrFail(pnh_, "regularization/nltgv2/init_with_prediction",
                   &params_.init_with_prediction);
    getParamOrFail(pnh_, "regularization/nltgv2/idepth_var_max",
                   &params_.idepth_var_max_graph);
    getParamOrFail(pnh_, "regularization/nltgv2/data_factor", &params_.rparams.data_factor);
    getParamOrFail(pnh_, "regularization/nltgv2/step_x", &params_.rparams.step_x);
    getParamOrFail(pnh_, "regularization/nltgv2/step_q", &params_.rparams.step_q);
    getParamOrFail(pnh_, "regularization/nltgv2/theta", &params_.rparams.theta);
    getParamOrFail(pnh_, "regularization/nltgv2/min_height", &params_.min_height);
    getParamOrFail(pnh_, "regularization/nltgv2/max_height", &params_.max_height);
    getParamOrFail(pnh_, "regularization/nltgv2/check_sticky_obstacles",
                   &params_.check_sticky_obstacles);

    /*==================== Analysis Params ====================*/
    getParamOrFail(pnh_, "analysis/pass_in_truth", &pass_in_truth_);

    // Setup input stream.
    FLAME_ASSERT(resize_factor_ == 1);
    input_ = std::make_shared<ros_sensor_streams::
    ASLRGBDOfflineStream>(nh_,
                          pose_path,
                          rgb_path,
                          depth_path,
                          "camera",
                          camera_world_frame_id_,
                          camera_frame_id_,
                          world_frame);

    // Set up publishers.
    if (publish_idepthmap_) {
        idepth_pub_ = it_.advertiseCamera("idepth_registered/image_rect", 5);
    }
    if (publish_depthmap_) {
        depth_pub_ = it_.advertiseCamera("depth_registered/image_rect", 5);
    }
    if (publish_features_) {
        features_pub_ = it_.advertiseCamera("depth_registered_raw/image_rect", 5);
    }
    if (publish_mesh_) {
        mesh_pub_ = nh_.advertise<pcl_msgs::PolygonMesh>("mesh", 5);
    }
    if (publish_cloud_) {
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cloud", 5);
    }


    if (params_.debug_draw_wireframe) {
        debug_wireframe_pub_ = it_.advertise("debug/wireframe", 1);
    }
    if (params_.debug_draw_features) {
        debug_features_pub_ = it_.advertise("debug/features", 1);
    }
    if (params_.debug_draw_detections) {
        debug_detections_pub_ = it_.advertise("debug/detections", 1);
    }
    if (params_.debug_draw_matches) {
        debug_matches_pub_ = it_.advertise("debug/matches", 1);
    }
    if (params_.debug_draw_normals) {
        debug_normals_pub_ = it_.advertise("debug/normals", 1);
    }
    if (params_.debug_draw_idepthmap) {
        debug_idepthmap_pub_ = it_.advertise("debug/idepthmap", 1);
    }

    return;
}

void DepthEstimate::estimate() {
    Kinv_ = input_->K().inverse();

    // Initialize depth sensor.
    ROS_INFO_COND(!params_.debug_quiet, "Constructing Flame...\n");
    sensor_ = std::make_shared<flame::Flame>(input_->width(),
                                             input_->height(),
                                             input_->K(),
                                             Kinv_,
                                             params_);

    /*==================== Enter main loop ====================*/
    ros::Rate ros_rate(rate_);
    ROS_INFO_COND(!params_.debug_quiet, "Done. We are GO for launch!\n");
    while (ros::ok() && !input_->empty()) {
        stats_.tick("main");

        // Get input data.
        stats_.set("queue_size", 0);
        stats_.tick("waiting");
        uint32_t img_id;
        double time;
        cv::Mat3b rgb;
        cv::Mat1f depth;
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
        input_->get(&img_id, &time, &rgb, &depth, &q, &t);
        stats_.tock("waiting");
        ROS_INFO_COND(!params_.debug_quiet,
                      "FlameNodelet/waiting = %4.1fms, queue_size = %i\n",
                      stats_.timings("waiting"),
                      static_cast<int>(stats_.stats("queue_size")));

        if (num_imgs_ % subsample_factor_ == 0) {
            // Eat data.
            Eigen::Isometry3d eigen_pose = Eigen::Isometry3d::Identity();
            eigen_pose.linear() = q.toRotationMatrix();
            eigen_pose.translation() = t;
            okvis::kinematics::Transformation pose(eigen_pose.matrix());


            processFrame(img_id, time, pose,
                         rgb, depth);
        }

        /*==================== Timing stuff ====================*/
        // Compute two measures of throughput in Hz. The first is the actual number of
        // frames per second, the second is the theoretical maximum fps based on the
        // runtime. They are not necessarily the same - the former takes external
        // latencies into account.

        // Compute maximum fps based on runtime.
        double fps_max = 0.0f;
        if (stats_.stats("fps_max") <= 0.0f) {
            fps_max = 1000.0f / stats_.timings("main");
        } else {
            fps_max = 1.0f / (0.99 * 1.0f/stats_.stats("fps_max") +
                              0.01 * stats_.timings("main")/1000.0f);
        }
        stats_.set("fps_max", fps_max);

        // Compute actual fps (overall throughput of system).
        stats_.tock("fps");
        double fps = 0.0;
        if (stats_.stats("fps") <= 0.0f) {
            fps = 1000.0f / stats_.timings("fps");
        } else {
            fps = 1.0f / (0.99 * 1.0f/stats_.stats("fps") +
                          0.01 * stats_.timings("fps")/1000.0f);
        }
        stats_.set("fps", fps);
        stats_.tick("fps");

        ros::spinOnce();
        ros_rate.sleep();

        stats_.tock("main");

        if ((num_imgs_ % load_integration_factor_) == 0) {
            // Compute load stats.
            fu::Load max_load, sys_load, pid_load;
            load_.get(&max_load, &sys_load, &pid_load);
            stats_.set("max_load_cpu", max_load.cpu);
            stats_.set("max_load_mem", max_load.mem);
            stats_.set("max_load_swap", max_load.swap);
            stats_.set("sys_load_cpu", sys_load.cpu);
            stats_.set("sys_load_mem", sys_load.mem);
            stats_.set("sys_load_swap", sys_load.swap);
            stats_.set("pid_load_cpu", pid_load.cpu);
            stats_.set("pid_load_mem", pid_load.mem);
            stats_.set("pid_load_swap", pid_load.swap);
            stats_.set("pid", getpid());
        }



        ROS_INFO_COND(!params_.debug_quiet,
                      "FlameOffline/main(%i/%u) = %4.1fms/%.1fHz (%.1fHz)\n",
                      num_imgs_, img_id, stats_.timings("main"),
                      stats_.stats("fps_max"), stats_.stats("fps"));

        num_imgs_++;
    }

    if (input_->empty()) {
        ROS_INFO("Finished processing.\n");
    } else {
        ROS_ERROR("Unknown error occurred!\n");
    }

    return;
}

void DepthEstimate::processFrame(const uint32_t img_id, const double time,
                  const okvis::kinematics::Transformation& pose, const cv::Mat3b& rgb,
                  const cv::Mat1f& depth) {
    stats_.tick("process_frame");

    /*==================== Process image ====================*/
    // Convert to grayscale.
    cv::Mat1b img_gray;
    cv::cvtColor(rgb, img_gray, cv::COLOR_RGB2GRAY);

    bool is_poseframe = (img_id % poseframe_subsample_factor_) == 0;
    bool update_success = false;
    if (!pass_in_truth_) {
        update_success = sensor_->update(time, img_id, pose, img_gray,
                                         is_poseframe);
    } else {
        // Convert true depth to idepth.
        cv::Mat1f idepths_true(rgb.rows, rgb.cols, std::numeric_limits<float>::quiet_NaN());
        for (int ii = 0; ii < depth.rows; ++ii) {
            for (int jj = 0; jj < depth.cols; ++jj) {
                if (!std::isnan(depth(ii, jj)) && (depth(ii, jj) > 0)) {
                    idepths_true(ii, jj) = 1.0f/depth(ii, jj);
                } else {
                    idepths_true(ii, jj) = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        update_success = sensor_->update(time, img_id, pose, img_gray, is_poseframe,
                                         idepths_true);
    }

    if (!update_success) {
        stats_.tock("process_frame");
        ROS_WARN("FlameOffline: Unsuccessful update.\n");
        return;
    }

    if (max_angular_rate_ > 0.0f) {
        // Check angle difference between last and current pose. If we're rotating,
        // we shouldn't publish output since it's probably too noisy.
        Eigen::Quaternionf q_delta = pose.hamilton_quaternion().cast<float>() *
                                     prev_pose_.hamilton_quaternion().inverse().cast<float>();
        float angle_delta = fu::fast_abs(Eigen::AngleAxisf(q_delta).angle());
        float angle_rate = angle_delta / (time - prev_time_);

        prev_time_ = time;
        prev_pose_ = pose;

        if (angle_rate * 180.0f / M_PI > max_angular_rate_) {
            // Angular rate is too high.
            ROS_ERROR_COND(!params_.debug_quiet,
                           "Angle Delta = %.3f, rate = %f.3\n", angle_delta * 180.0f / M_PI,
                           angle_rate * 180.0f / M_PI);
            return;
        }
    }

    /*==================== Publish output ====================*/
    stats_.tick("publishing");

    if (publish_mesh_) {
        // Get current mesh.
        std::vector<cv::Point2f> vtx;
        std::vector<float> idepths;
        std::vector<Eigen::Vector3f> normals;
        std::vector<flame::Triangle> triangles;
        std::vector<flame::Edge> edges;
        std::vector<bool> tri_validity;
        sensor_->getInverseDepthMesh(&vtx, &idepths, &normals, &triangles,
                                     &tri_validity, &edges);
        publishDepthMesh(mesh_pub_, camera_frame_id_, time, Kinv_, vtx,
                         idepths, normals, triangles, tri_validity, rgb);
    }

    cv::Mat1f idepthmap;
    if (publish_idepthmap_ || publish_depthmap_ || publish_cloud_) {
        cv::Mat1f idepthmap;
        sensor_->getFilteredInverseDepthMap(&idepthmap);

        if (publish_idepthmap_) {
            publishDepthMap(idepth_pub_, camera_frame_id_, time, input_->K(),
                            idepthmap);
        }

        // Convert to depths.
        cv::Mat1f depth_est(idepthmap.rows, idepthmap.cols,
                            std::numeric_limits<float>::quiet_NaN());
#pragma omp parallel for collapse(2) num_threads(params_.omp_num_threads) schedule(dynamic, params_.omp_chunk_size) // NOLINT
        for (int ii = 0; ii < depth_est.rows; ++ii) {
            for (int jj = 0; jj < depth_est.cols; ++jj) {
                float idepth =  idepthmap(ii, jj);
                if (!std::isnan(idepth) && (idepth > 0)) {
                    depth_est(ii, jj) = 1.0f/ idepth;
                }
            }
        }

        if (publish_depthmap_) {
            publishDepthMap(depth_pub_, camera_frame_id_, time, input_->K(),
                            depth_est);
        }

        if (publish_cloud_) {
            float max_depth = (params_.do_idepth_triangle_filter) ?
                              1.0f / params_.min_triangle_idepth : std::numeric_limits<float>::max();
            publishPointCloud(cloud_pub_, camera_frame_id_, time, input_->K(),
                              depth_est, 0.1f, max_depth);
        }
    }

    if (publish_features_) {
        cv::Mat1f depth_raw(img_gray.rows, img_gray.cols,
                            std::numeric_limits<float>::quiet_NaN());
        if (publish_features_) {
            std::vector<cv::Point2f> vertices;
            std::vector<float> idepths_mu, idepths_var;
            sensor_->getRawIDepths(&vertices, &idepths_mu, &idepths_var);

            for (int ii = 0; ii < vertices.size(); ++ii) {
                float id = idepths_mu[ii];
                float var = idepths_var[ii];
                if (!std::isnan(id) && (id > 0)) {
                    int x = fu::fast_roundf(vertices[ii].x);
                    int y = fu::fast_roundf(vertices[ii].y);

                    FLAME_ASSERT(x >= 0);
                    FLAME_ASSERT(x < depth_raw.cols);
                    FLAME_ASSERT(y >= 0);
                    FLAME_ASSERT(y < depth_raw.rows);

                    depth_raw(y, x) = 1.0f / id;
                }
            }
        }

        publishDepthMap(features_pub_, camera_frame_id_, time, input_->K(),
                        depth_raw);
    }



    stats_.set("latency", (ros::Time::now().toSec() - time) * 1000);
    ROS_INFO_COND(!params_.debug_quiet,
                  "FlameNodelet/latency = %4.1fms\n",
                  stats_.stats("latency"));

    stats_.tock("publishing");
    ROS_INFO_COND(!params_.debug_quiet,
                  "FlameOffline/publishing = %4.1fms\n",
                  stats_.timings("publishing"));

    /*==================== Publish debug stuff ====================*/
    stats_.tick("debug_publishing");

    std_msgs::Header hdr;
    hdr.stamp.fromSec(time);
    hdr.frame_id = camera_frame_id_;

    if (params_.debug_draw_wireframe) {
        sensor_msgs::Image::Ptr debug_img_msg =
                cv_bridge::CvImage(hdr, "bgr8",
                                   sensor_->getDebugImageWireframe()).toImageMsg();
        debug_wireframe_pub_.publish(debug_img_msg);
    }

    if (params_.debug_draw_features) {
        sensor_msgs::Image::Ptr debug_img_msg =
                cv_bridge::CvImage(hdr, "bgr8",
                                   sensor_->getDebugImageFeatures()).toImageMsg();
        debug_features_pub_.publish(debug_img_msg);
    }

    if (params_.debug_draw_detections) {
        sensor_msgs::Image::Ptr debug_img_msg =
                cv_bridge::CvImage(hdr, "bgr8",
                                   sensor_->getDebugImageDetections()).toImageMsg();
        debug_detections_pub_.publish(debug_img_msg);
    }

    if (params_.debug_draw_matches) {
        sensor_msgs::Image::Ptr debug_img_msg =
                cv_bridge::CvImage(hdr, "bgr8",
                                   sensor_->getDebugImageMatches()).toImageMsg();
        debug_matches_pub_.publish(debug_img_msg);
    }

    if (params_.debug_draw_normals) {
        sensor_msgs::Image::Ptr debug_img_msg =
                cv_bridge::CvImage(hdr, "bgr8",
                                   sensor_->getDebugImageNormals()).toImageMsg();
        debug_normals_pub_.publish(debug_img_msg);
    }

    if (params_.debug_draw_idepthmap) {
        sensor_msgs::Image::Ptr debug_img_msg =
                cv_bridge::CvImage(hdr, "bgr8",
                                   sensor_->getDebugImageInverseDepthMap()).toImageMsg();
        debug_idepthmap_pub_.publish(debug_img_msg);
    }

    stats_.tock("debug_publishing");
    ROS_INFO_COND(!params_.debug_quiet,
                  "FlameOffline/debug_publishing = %4.1fms\n",
                  stats_.timings("debug_publishing"));

    stats_.tock("process_frame");

    ROS_INFO_COND(!params_.debug_quiet,
                  "FlameOffline/process_frame = %4.1fms\n",
                  stats_.timings("process_frame"));

    return;
}


