#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <common/Time/Time.hpp>
#include <common/measurements.h>
#include <common/threadsafe/ThreadsafeQueue.hpp>
#include <common/ImuFrameSynchronizer.hpp>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/pipeline/visual-npipeline.h>
#include <aslam/pipeline/visual-pipeline-null.h>
#include <feature-tracking/vo-feature-tracking-pipeline.h>

#include <depth_mesh/mesh_estimator.hpp>
#include <depth_mesh/params.h>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


struct StereoFeatures {
    okvis::Time timeStamp;
    okvis::StereoCameraData stereoCameraData;
    std::map<int, vector<pair<int, Vector3d>>> features;
};


struct UpdateMeshInfo {
    std::vector<State> vio_state;
    bool isKeyframe;
    okvis::StereoCameraData stereoCameraData;
};

okvis::ImuFrameSynchronizer imuFrameSynchronizer;
Estimator estimator;


okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement> imuThreadSafeQueue;

okvis::threadsafe::ThreadSafeQueue<okvis::StereoCameraMeasurement> stereoCameraThreadSafeQueue;

okvis::threadsafe::ThreadSafeQueue<std::pair<okvis::ImuMeasurementDeque,
                                   okvis::StereoCameraMeasurement>> imuImagesPackageThreadSafeQueue;

okvis::threadsafe::ThreadSafeQueue<std::pair<okvis::ImuMeasurementDeque,
        StereoFeatures>> imuFeaturesPackageThreadSafeQueue;



std::shared_ptr<aslam::NCamera> camera_system;
std::shared_ptr<feature_tracking::VOFeatureTrackingPipeline> voFeatureTrackingPipeline;
aslam::VisualNPipeline::UniquePtr visual_pipeline;

std::shared_ptr<aslam::VisualNFrame> nframe_k, nframe_kp1;
aslam::FrameToFrameMatchesList  inlier_matches_kp1_k;
aslam::FrameToFrameMatchesList  outlier_matches_kp1_k;

bool feature_tracking_avalible = false;
std::mutex inliner_mutex;
cv::Mat inliner_view0_ft;
cv::Mat inliner_view1_ft;


///< Mesh estimate
Eigen::Matrix3d K0, K1;
Eigen::Vector4d distort0, distort1;
int imageWidth, imageHeight;

flame::Params mesh_est_param;
//std::shared_ptr<flame::MeshEstimator> mesh_estimator;
okvis::threadsafe::ThreadSafeQueue<UpdateMeshInfo> meshUpdateInfoThreadSafeQueue;




std::condition_variable con;
double current_time = -1;
queue<okvis::ImuMeasurement> imu_buf;

ros::Publisher pub_img,pub_match, pub_depth;

int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_loop_drift;


double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;


//camera param
std_msgs::Header cur_header;
Eigen::Vector3d relocalize_t{Eigen::Vector3d(0, 0, 0)};
Eigen::Matrix3d relocalize_r{Eigen::Matrix3d::Identity()};


void predict(const okvis::ImuMeasurement &imu_msg)
{
    double t = imu_msg.timeStamp.toSec();
    double dt = t - latest_time;
    latest_time = t;

    Eigen::Vector3d linear_acceleration = imu_msg.measurement.accelerometers;
    Eigen::Vector3d angular_velocity = imu_msg.measurement.gyroscopes;


    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba - tmp_Q.inverse() * estimator.g);

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba - tmp_Q.inverse() * estimator.g);

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = relocalize_r * estimator.Ps[WINDOW_SIZE] + relocalize_t;
    tmp_Q = relocalize_r * estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<okvis::ImuMeasurement> tmp_imu_buf = imu_buf;
    for (okvis::ImuMeasurement tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

void send_imu(const okvis::ImuMeasurement &imu_msg)
{
    double t = imu_msg.timeStamp.toSec();
    if (current_time < 0)
        current_time = t;
    double dt = t - current_time;
    current_time = t;

    double ba[]{0.0, 0.0, 0.0};
    double bg[]{0.0, 0.0, 0.0};

    estimator.processIMU(dt, imu_msg.measurement.accelerometers, imu_msg.measurement.gyroscopes);
}


// Get a subset of the recorded IMU measurements.
okvis::ImuMeasurementDeque getImuMeasurments(
        okvis::Time& imuDataBeginTime, okvis::Time& imuDataEndTime) {
    okvis::ImuMeasurementDeque sub_imu_measurements;
    while(!imuThreadSafeQueue.Empty()) {
        okvis::ImuMeasurement imuMeasurement;
        imuThreadSafeQueue.getCopyOfFront(&imuMeasurement);
        if (imuMeasurement.timeStamp < imuDataBeginTime)
            imuThreadSafeQueue.PopNonBlocking(&imuMeasurement);
        else if (imuMeasurement.timeStamp <= imuDataEndTime) {
            imuThreadSafeQueue.PopNonBlocking(&imuMeasurement);
            sub_imu_measurements.push_back(imuMeasurement);
        } else {
            break;
        }
    }
    return sub_imu_measurements;
}

std::shared_ptr<aslam::NCamera> loadCameraInfoFromConfigFile(
        const std::string& config_file) {
    aslam::NCamera::Ptr camera_system = nullptr;
    cv::FileStorage fs(config_file, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        return camera_system;
    }

    imageWidth = static_cast<int>(fs["image_width"]);
    imageHeight = static_cast<int>(fs["image_height"]);


    // For camera 0
    cv::FileNode n = fs["distortion_parameters"];
    double k1 = static_cast<double>(n["k1"]);
    double k2 = static_cast<double>(n["k2"]);
    double p1 = static_cast<double>(n["p1"]);
    double p2 = static_cast<double>(n["p2"]);

    n = fs["projection_parameters"];
    double fx = static_cast<double>(n["fx"]);
    double fy = static_cast<double>(n["fy"]);
    double cx = static_cast<double>(n["cx"]);
    double cy = static_cast<double>(n["cy"]);

    cv::Mat cv_R, cv_T;
    fs["extrinsicRotation"] >> cv_R;
    fs["extrinsicTranslation"] >> cv_T;

    K0 = Eigen::Matrix3d::Identity();
    K0(0,0) = fx; K0(1,1) = fy; K0(0,2) = cx; K0(1,2) = cy;
    distort0 << k1,k2,p1,p2;


    Eigen::Matrix3d eigen_R;
    Eigen::Vector3d eigen_T;
    cv::cv2eigen(cv_R, eigen_R);
    cv::cv2eigen(cv_T, eigen_T);

    std::cout<<"eigen_R: " << eigen_R << std::endl;
    std::cout<<"eigen_T: " << eigen_T << std::endl;

    Eigen::VectorXd dist_coeffs(4);
    dist_coeffs << k1, k2, p1, p2;
    aslam::Distortion::UniquePtr distortion =
            std::unique_ptr<aslam::RadTanDistortion>(
                    new aslam::RadTanDistortion(dist_coeffs));
    std::cout<<"dist_coeffs: "<<dist_coeffs<<std::endl;

    Eigen::VectorXd cameraIntrinsic(4);
    cameraIntrinsic << fx, fy, cx, cy;
    std::cout<<"cameraIntrinsic: "<<cameraIntrinsic<<std::endl;

    aslam::Camera::Ptr camera = std::shared_ptr<aslam::PinholeCamera>(
            new aslam::PinholeCamera(
                    cameraIntrinsic, imageWidth, imageHeight, distortion));
    aslam::CameraId cameraId;
    cameraId.randomize();
    camera->setId(cameraId);

    Eigen::Quaterniond eigQ_BC(eigen_R);
    kindr::minimal::RotationQuaternionTemplate<double> q_BC(
            eigQ_BC.w(), eigQ_BC.x(), eigQ_BC.y(), eigQ_BC.z());
    Eigen::Vector3d t_BC;
    t_BC = eigen_T;

    aslam::Transformation T_CB(
            q_BC.conjugated(), -q_BC.conjugated().rotate(t_BC));

    // For camera 1

    n = fs["distortion_parameters1"];
    k1 = static_cast<double>(n["k1"]);
    k2 = static_cast<double>(n["k2"]);
    p1 = static_cast<double>(n["p1"]);
    p2 = static_cast<double>(n["p2"]);

    n = fs["projection_parameters1"];
    fx = static_cast<double>(n["fx"]);
    fy = static_cast<double>(n["fy"]);
    cx = static_cast<double>(n["cx"]);
    cy = static_cast<double>(n["cy"]);


    fs["extrinsicRotation1"] >> cv_R;
    fs["extrinsicTranslation1"] >> cv_T;

    K1 = Eigen::Matrix3d::Identity();
    K1(0,0) = fx; K1(1,1) = fy; K1(0,2) = cx; K1(1,2) = cy;
    distort1 << k1,k2,p1,p2;


    Eigen::Matrix3d eigen_R1;
    Eigen::Vector3d eigen_T1;
    cv::cv2eigen(cv_R, eigen_R1);
    cv::cv2eigen(cv_T, eigen_T1);

    std::cout<<"eigen_R1: " << eigen_R1 << std::endl;
    std::cout<<"eigen_T1: " << eigen_T1 << std::endl;

    // RadTanDistortion

    Eigen::VectorXd dist_coeffs1(4);
    dist_coeffs1 << k1, k2, p1, p2;
    aslam::Distortion::UniquePtr distortion1 =
            std::unique_ptr<aslam::RadTanDistortion>(
                    new aslam::RadTanDistortion(dist_coeffs1));
    std::cout<<"dist_coeffs1: "<<dist_coeffs1<<std::endl;


    Eigen::VectorXd cameraIntrinsic1(4);
    cameraIntrinsic1 << fx, fy, cx, cy;
    std::cout<<"cameraIntrinsic1: "<<cameraIntrinsic1<<std::endl;

    aslam::Camera::Ptr camera1 = std::shared_ptr<aslam::PinholeCamera>(
            new aslam::PinholeCamera(
                    cameraIntrinsic1, imageWidth, imageHeight, distortion1));
    aslam::CameraId cameraId1;
    cameraId1.randomize();
    camera1->setId(cameraId1);

    Eigen::Quaterniond eigQ_BC1(eigen_R1);
    kindr::minimal::RotationQuaternionTemplate<double> q_BC1(
            eigQ_BC1.w(), eigQ_BC1.x(), eigQ_BC1.y(), eigQ_BC1.z());
    Eigen::Vector3d t_BC1;
    t_BC1 = eigen_T1;

    aslam::Transformation T_CB1(
            q_BC1.conjugated(), -q_BC1.conjugated().rotate(t_BC1));


    std::string label = "cam0";
    aslam::TransformationVector tv;
    tv.push_back(T_CB);
    tv.push_back(T_CB1);
    std::vector<std::shared_ptr<aslam::Camera>> vCamera;
    vCamera.push_back(camera);
    vCamera.push_back(camera1);

    aslam::NCameraId id;
    id.randomize();

    camera_system = std::shared_ptr<aslam::NCamera>(
            new aslam::NCamera(id, tv, vCamera, label));

    fs.release();
    return camera_system;
}

void integrateInterframeImuRotation(
        const okvis::ImuMeasurementDeque& imu_measurements,
        const Eigen::Vector3d gyro_bias,
        aslam::Quaternion* q_Ikp1_Ik)  {
    okvis::ImuMeasurementDeque imu_queue =  imu_measurements;

    q_Ikp1_Ik->setIdentity();
    double last_ts = imu_queue.front().timeStamp.toSec();
    // pop first
    imu_queue.pop_front();
    for (;  !imu_queue.empty();) {
        const double delta_s =
                imu_queue.front().timeStamp.toSec() - last_ts;
        CHECK_GT(delta_s, 0);
        const Eigen::Vector3d gyro_measurement
        = imu_queue.front().measurement.gyroscopes;

        *q_Ikp1_Ik =
                *q_Ikp1_Ik * aslam::Quaternion::exp(gyro_measurement * delta_s);
        last_ts = imu_queue.front().timeStamp.toSec();
        imu_queue.pop_front();
    }
    // We actually need to inverse the rotation so that transform from Ikp1 to Ik.
    *q_Ikp1_Ik = q_Ikp1_Ik->inverse();
}

okvis::Time last_images_ts = okvis::Time(0.0);
void synchronize() {
    for (;;) {
        okvis::StereoCameraMeasurement stereoCameraMeasurement;
        if (stereoCameraThreadSafeQueue.PopBlocking(&stereoCameraMeasurement) == false)
            return;

        if (!imuFrameSynchronizer.waitForUpToDateImuData(stereoCameraMeasurement.timeStamp) )
            return;

        if (stereoCameraMeasurement.timeStamp.toSec() - last_images_ts.toSec() > 0.1 ) {
            double dt  = stereoCameraMeasurement.timeStamp.toSec() - 0.1;
            last_images_ts = okvis::Time(dt);
        }

        okvis::ImuMeasurementDeque imuMeasurementDeque
                            = getImuMeasurments(last_images_ts, stereoCameraMeasurement.timeStamp);

        std::pair<okvis::ImuMeasurementDeque,
                okvis::StereoCameraMeasurement> imuImagesPackage
                = std::make_pair(imuMeasurementDeque, stereoCameraMeasurement);
        imuImagesPackageThreadSafeQueue.PushNonBlockingDroppingIfFull(imuImagesPackage, 10);
    }
}

void track_featrues() {
    bool  first_frame  =  true;
    for (;;) {
        std::pair<okvis::ImuMeasurementDeque,
                okvis::StereoCameraMeasurement> imuImagesPackage;
        if (imuImagesPackageThreadSafeQueue.PopBlocking(&imuImagesPackage) == false)
            return;

        visual_pipeline->processImageBlockingIfFull(
                0, imuImagesPackage.second.measurement.image0,
                imuImagesPackage.second.timeStamp.toNSec(), 10);
        visual_pipeline->processImageBlockingIfFull(
                1, imuImagesPackage.second.measurement.image0,
                imuImagesPackage.second.timeStamp.toNSec(), 10);

        aslam::VisualNFrame::Ptr new_nframe;
        visual_pipeline->getNextBlocking(&new_nframe);
        std::cout<< " new_nframe: " << new_nframe->getNumCameras() <<" "
                     << new_nframe->getMaxTimestampNanoseconds() << std::endl;


        nframe_kp1 = new_nframe;
        if(!first_frame) {
            aslam::Quaternion aslam_q;
            Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
            integrateInterframeImuRotation(imuImagesPackage.first, gyro_bias, &aslam_q);

            TicToc timing;
            voFeatureTrackingPipeline->trackFeaturesNFrame(aslam_q,nframe_kp1.get(), nframe_k.get(),
                                                           &inlier_matches_kp1_k,
                                                           &outlier_matches_kp1_k);

            std::cout<< "Timing: " << timing.toc() << " " << inlier_matches_kp1_k[0].size() << std::endl;

//            inliner_mutex.lock();
//            feature_tracking_avalible = true;
//            inliner_view0_ft = voFeatureTrackingPipeline->inliner_view0;
//            inliner_view1_ft = voFeatureTrackingPipeline->inliner_view1;
//            inliner_mutex.unlock();

            cv_bridge::CvImage out_msg;
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = "world";
            out_msg.header   = header; // Same timestamp and tf frame as input image
            out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
            out_msg.image    = voFeatureTrackingPipeline->inliner_view0; // Your cv::Mat


            pub_match.publish(out_msg.toImageMsg());




//        std::cout<< "------" << std::endl;
//            aslam::FrameToFrameMatches frame0_matches = inlier_matches_kp1_k[0];
//        for (auto match: frame0_matches) {
//          std::cout<< "match: " << frame0_matches.size() << " "
//          << match.second << " " << match.first << " "
//          << nframe_kp1->getFrame(0).getTrackId(match.first)<< " "
//          << nframe_k->getFrame(0).getTrackId(match.second) <<  std::endl;
//        }

            std::map<int, vector<pair<int, Vector3d>>> image;
            for (auto match : inlier_matches_kp1_k[0]) {
                size_t matched_index = match.first;
                int tracked_id = nframe_kp1->getFrame(0).getTrackId(matched_index);
                const Eigen::Vector2d& keypoint
                        = nframe_kp1->getFrame(0).getKeypointMeasurement(matched_index);
                Eigen::Vector3d back_project;
                camera_system->getCamera(0).backProject3(keypoint, &back_project);
                Eigen::Vector3d bearing(back_project.x() / back_project.z(),
                        back_project.y() / back_project.z(),
                        1.0);
                //std::cout<< "tracked_id: " << tracked_id << " " << bearing.transpose() << std::endl;
                image[tracked_id].emplace_back(0, bearing);
            }

            StereoFeatures stereoFeatures;
            stereoFeatures.timeStamp = imuImagesPackage.second.timeStamp;
            stereoFeatures.features = image;
            stereoFeatures.stereoCameraData = imuImagesPackage.second.measurement;

            std::pair<okvis::ImuMeasurementDeque, StereoFeatures>
                    imuFeaturesPackage = std::make_pair(imuImagesPackage.first, stereoFeatures);



            // Add into buffer
            imuFeaturesPackageThreadSafeQueue.PushNonBlockingDroppingIfFull(imuFeaturesPackage, 10);

        } else {
            first_frame = false;
        }


        // Update
        nframe_k = nframe_kp1;
    }
}

void vio_estimate() {
    for (;;) {
        std::pair<okvis::ImuMeasurementDeque,
                        StereoFeatures> imuFeaturesPackage;
        if (imuFeaturesPackageThreadSafeQueue.PopBlocking(&imuFeaturesPackage) == false)
            return;


        // Add imu
        while (! imuFeaturesPackage.first.empty()) {
            okvis::ImuMeasurement imu_msg =  imuFeaturesPackage.first.front();
            send_imu(imu_msg);
            imuFeaturesPackage.first.pop_front();
        }


        TicToc t_s;
        std_msgs::Header header;
        header.stamp = ros::Time(imuFeaturesPackage.second.timeStamp.toSec());
        estimator.processImage(imuFeaturesPackage.second.features, header);
        double whole_t = t_s.toc();

        std::cout<< "processImage: " << whole_t << std::endl;

        printStatistics(estimator, whole_t);
        header.frame_id = "world";
        cur_header = header;
        m_loop_drift.lock();
        if (estimator.relocalize)
        {
            relocalize_t = estimator.relocalize_t;
            relocalize_r = estimator.relocalize_r;
        }
        pubOdometry(estimator, header, relocalize_t, relocalize_r);
        pubKeyPoses(estimator, header, relocalize_t, relocalize_r);
        pubCameraPose(estimator, header, relocalize_t, relocalize_r);
        pubPointCloud(estimator, header, relocalize_t, relocalize_r);
        pubTF(estimator, header, relocalize_t, relocalize_r);
        m_loop_drift.unlock();
        //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());

        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();


        // Add result into buffer

        if (estimator.isInitalized()) {
            bool isKeyframe = estimator.isKeyframe();
            std::vector<State> states = estimator.getCurrentStates();
            UpdateMeshInfo updateMeshInfo;
            updateMeshInfo.isKeyframe = isKeyframe;
            updateMeshInfo.vio_state = states;
            updateMeshInfo.stereoCameraData = imuFeaturesPackage.second.stereoCameraData;


            meshUpdateInfoThreadSafeQueue.PushNonBlockingDroppingIfFull(updateMeshInfo, 10);
        }

    }

}

void estimate_depth_mesh() {
    uint64_t id = 0;
    for (;;) {
        UpdateMeshInfo updateMeshInfo;
        if (meshUpdateInfoThreadSafeQueue.PopBlocking(&updateMeshInfo) == false)
            return;

        // todo
        std::cout<< "get meshUpdate info "  << updateMeshInfo.vio_state.size()
        << " " << updateMeshInfo.isKeyframe <<  std::endl;

        Eigen::Isometry3d eigen_T_WS = Eigen::Isometry3d::Identity();
        eigen_T_WS.linear() = updateMeshInfo.vio_state.back().R;
        eigen_T_WS.translation() = updateMeshInfo.vio_state.back().P;
        okvis::kinematics::Transformation T_WS(eigen_T_WS.matrix());

        aslam::Transformation aslam_T_B_C0= camera_system->get_T_C_B(0).inverse();
        Eigen::Isometry3d T_BC0 = Eigen::Isometry3d::Identity();
        T_BC0.translation() = aslam_T_B_C0.getPosition();
        T_BC0.linear() = aslam_T_B_C0.getRotation().toImplementation().toRotationMatrix();
        okvis::kinematics::Transformation T_WC0 = T_WS * okvis::kinematics::Transformation(T_BC0.matrix());

//        std::cout<<"T_BC0: " << std::endl << T_BC0.matrix();

        aslam::Transformation aslam_T_B_C1= camera_system->get_T_C_B(1).inverse();
        Eigen::Isometry3d T_BC1 = Eigen::Isometry3d::Identity();
        T_BC1.translation() = aslam_T_B_C1.getPosition();
        T_BC1.linear() = aslam_T_B_C1.getRotation().toImplementation().toRotationMatrix();
        okvis::kinematics::Transformation T_WC1 = T_WS * okvis::kinematics::Transformation(T_BC1.matrix());

//        std::cout<<"T_BC1: " << std::endl << T_BC1.matrix();


        okvis::Time time(updateMeshInfo.vio_state.back().Header.stamp.toSec());
        bool isKeyframe = updateMeshInfo.isKeyframe;

//        mesh_estimator->processFrame(time, id++,
//                                        T_WC0, updateMeshInfo.stereoCameraData.image0,
//                                        T_WC1, updateMeshInfo.stereoCameraData.image1,
//                                        isKeyframe);
//
//        const flame::Image3b depth_image =  mesh_estimator->getDebugImageWireframe();
//
//
//        cv_bridge::CvImage out_msg;
//        std_msgs::Header header;
//        header.stamp = ros::Time::now();
//        header.frame_id = "world";
//        out_msg.header   = header; // Same timestamp and tf frame as input image
//        out_msg.encoding = sensor_msgs::image_encodings::RGB8; // Or whatever
//        out_msg.image    = depth_image; // Your cv::Mat
//
//
//        pub_depth.publish(out_msg.toImageMsg());


    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");



    registerPub(n);
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_depth = n.advertise<sensor_msgs::Image>("depth_img",1000);


    std::string config_file
        = "/home/pang/maplab_ws/src/vio_mesh_estimate/config/euroc/euroc_config.yaml";

    camera_system = loadCameraInfoFromConfigFile(config_file);
    voFeatureTrackingPipeline
        = std::make_shared<feature_tracking::VOFeatureTrackingPipeline>(camera_system);


    // Initialize the pipeline.
    static constexpr bool kCopyImages = false;
    std::vector<aslam::VisualPipeline::Ptr> mono_pipelines;
    for (size_t camera_idx = 0; camera_idx < camera_system->getNumCameras();
         ++camera_idx) {
        mono_pipelines.emplace_back(
                new aslam::NullVisualPipeline(
                        camera_system->getCameraShared(camera_idx), kCopyImages));
    }

    constexpr size_t kNumThreads = 1u;
    const int kNFrameToleranceNs = 500000;
    visual_pipeline.reset(new aslam::VisualNPipeline(
            kNumThreads, mono_pipelines, camera_system, camera_system,
            kNFrameToleranceNs));


//    const Eigen::VectorXd cam0_intrinsic = camera_system->getCamera(0).getParameters();
//    const Eigen::VectorXd cam1_intrinsic = camera_system->getCamera(1).getParameters();

//    mesh_estimator = std::make_shared<flame::MeshEstimator>(imageWidth, imageHeight,
//                                             K0.cast<float>(), K0.inverse().cast<float>(),
//                                             distort0.cast<float>(),
//                                             K1.cast<float>(), K1.inverse().cast<float>(),
//                                             distort1.cast<float>(),
//                                             mesh_est_param);


    std::thread imu_images_synchronize{synchronize};
    std::thread feature_tracking{track_featrues};
    std::thread estimate{vio_estimate};
    std::thread depth_mesh_estimate{estimate_depth_mesh};

    /**
     *  Date Source
     */
    // the folder path
    std::string path =  "/home/pang/dataset/euroc/MH_01_easy/mav0";

    okvis::Duration deltaT(0.0);


    // open the IMU file
    std::string line;
    std::ifstream imu_file(path + "/imu0/data.csv");
    if (!imu_file.good()) {
        LOG(ERROR)<< "no imu file found at " << path+"/imu0/data.csv";
        return -1;
    }
    int number_of_lines = 0;
    while (std::getline(imu_file, line))
        ++number_of_lines;
    LOG(INFO)<< "No. IMU measurements: " << number_of_lines-1;
    if (number_of_lines - 1 <= 0) {
        LOG(ERROR)<< "no imu messages present in " << path+"/imu0/data.csv";
        return -1;
    }
    // set reading position to second line
    imu_file.clear();
    imu_file.seekg(0, std::ios::beg);
    std::getline(imu_file, line);

    int numCameras = 2;
    std::vector<okvis::Time> times;
    okvis::Time latest(0);
    int num_camera_images = 0;
    std::vector < std::vector < std::string >> image_names(numCameras);
    for (size_t i = 0; i < numCameras; ++i) {
        num_camera_images = 0;
        std::string folder(path + "/cam" + std::to_string(i) + "/data");

        for (auto it = boost::filesystem::directory_iterator(folder);
             it != boost::filesystem::directory_iterator(); it++) {
            if (!boost::filesystem::is_directory(it->path())) {  //we eliminate directories
                num_camera_images++;
                image_names.at(i).push_back(it->path().filename().string());
            } else {
                continue;
            }
        }

        if (num_camera_images == 0) {
            LOG(ERROR)<< "no images at " << folder;
            return 1;
        }

        LOG(INFO)<< "No. cam " << i << " images: " << num_camera_images;
        // the filenames are not going to be sorted. So do this here
        std::sort(image_names.at(i).begin(), image_names.at(i).end());
    }

    std::vector < std::vector < std::string > ::iterator
    > cam_iterators(numCameras);
    for (size_t i = 0; i < numCameras; ++i) {
        cam_iterators.at(i) = image_names.at(i).begin();
    }

    double last_ts;
    bool first_frame = true;
    std::chrono::steady_clock::time_point cur_moment, last_moment;

    int counter = 0;
    okvis::Time start(0.0);
    while (ros::ok()) {
        // check if at the end
        for (size_t i = 0; i < numCameras; ++i) {
            if (cam_iterators[i] == image_names[i].end()) {
                std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
                cv::waitKey();
                return 0;
            }
        }

        /// add images
        okvis::Time t;

        cv::Mat filtered0 = cv::imread(
                path + "/cam" + std::to_string(0) + "/data/" + *cam_iterators.at(0),
                cv::IMREAD_GRAYSCALE);
        cv::Mat filtered1 = cv::imread(
                path + "/cam" + std::to_string(1) + "/data/" + *cam_iterators.at(1),
                cv::IMREAD_GRAYSCALE);


        std::string nanoseconds = cam_iterators.at(0)->substr(
                cam_iterators.at(0)->size() - 13, 9);
        std::string seconds = cam_iterators.at(0)->substr(
                0, cam_iterators.at(0)->size() - 13);
        t = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));
        if (start == okvis::Time(0.0)) {
            start = t;
        }

        // get all IMU measurements till then
        okvis::Time t_imu = start;
        do {
            if (!std::getline(imu_file, line)) {
                std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
                cv::waitKey();
                return 0;
            }

            std::stringstream stream(line);
            std::string s;
            std::getline(stream, s, ',');
            std::string nanoseconds = s.substr(s.size() - 9, 9);
            std::string seconds = s.substr(0, s.size() - 9);

            Eigen::Vector3d gyr;
            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');
                gyr[j] = std::stof(s);
            }

            Eigen::Vector3d acc;
            for (int j = 0; j < 3; ++j) {
                std::getline(stream, s, ',');
                acc[j] = std::stof(s);
            }

            t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));

            // add the IMU measurement for (blocking) processing
            if (t_imu - start + okvis::Duration(1.0) > deltaT) {
                okvis::ImuSensorReadings imuSensorReadings(gyr, acc);
                okvis::ImuMeasurement imuMeasurement(t_imu, imuSensorReadings);

                imuThreadSafeQueue.PushNonBlockingDroppingIfFull(imuMeasurement, 200);
                imuFrameSynchronizer.gotImuData(imuMeasurement.timeStamp);


            }

        } while (t_imu <= t);

        // add the image to the frontend for (blocking) processing
        if (t - start > deltaT) {
            okvis::CameraData cameraData;
            cameraData.image = filtered0;
            okvis::CameraMeasurement cameraMeasurement(t, cameraData);

            okvis::StereoCameraData stereoCameraData;
            stereoCameraData.image0 = filtered0;
            stereoCameraData.image1 = filtered1;
            okvis::StereoCameraMeasurement stereoCameraMeasurement(t, stereoCameraData);

            stereoCameraThreadSafeQueue.PushNonBlockingDroppingIfFull(stereoCameraMeasurement, 10);


            // Compensate real-world timing slips
            if (first_frame) {
                last_moment = std::chrono::steady_clock::now();
                first_frame = false;
            } else {
                cur_moment = std::chrono::steady_clock::now();
                double spend =
                        std::chrono::duration_cast<std::chrono::duration<double>>(
                                cur_moment - last_moment)
                                .count();
                double target = cameraMeasurement.timeStamp.toSec() - last_ts;

                if (spend < target) {
                    double need_sleep = target - spend;
                    usleep((need_sleep)*1e6);
                }
                last_moment = std::chrono::steady_clock::now();
            }

            last_ts =  cameraMeasurement.timeStamp.toSec();
        }

        cam_iterators[0]++;
        cam_iterators[1]++;

        ++counter;


//        inliner_mutex.lock();
//        bool availible = feature_tracking_avalible;
//
//        cv::Mat inliner_view0 = inliner_view0_ft;
//        cv::Mat inliner_view1 = inliner_view1_ft;
//        inliner_mutex.unlock();
//
//        if (availible) {
//            cv::imshow("inliner 0", inliner_view0);
//            //cv::imshow("inliner 1", inliner_view1);
//            cv::waitKey(1);
//        }


        // display progress
        if (counter % 20 == 0) {
            std::cout << "\rProgress: "
                      << int(double(counter) / double(num_camera_images) * 100) << "%  "
                      << std::flush;
        }
        ros::spinOnce();
    }
    return 0;
}
