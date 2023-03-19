#include <ros/ros.h>
#include <tf2/buffer_core.h>
#include <tf2_client/tf2_client.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/buffer_client.h>
#include <tf2_ros/transform_listener.h>

namespace tf2_client {

BufferPtr get_buffer(ros::NodeHandle &nh, ros::NodeHandle &pnh) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lck(mtx);
    static BufferPtr buffer;
    typedef std::unique_ptr<tf2_ros::TransformListener> ListenerPtr;
    static ListenerPtr listener;
    if (!buffer) {
        std::string server;
        pnh.param("tf_server", server, server);
        if (!server.empty()) {
            double check_frequency(10.0);
            pnh.param("tf_check_frequency", check_frequency, check_frequency);
            double timeout_padding(2.0);
            pnh.param("tf_timeout_padding", timeout_padding, timeout_padding);

            buffer = std::make_shared<tf2_ros::BufferClient>(server, check_frequency, ros::Duration(timeout_padding));
            ROS_INFO("Using tf buffer client (server %s, timeout padding %.3g s).", server.c_str(), timeout_padding);
        } else {
            double cache_time(tf2::BufferCore::DEFAULT_CACHE_TIME);
            pnh.param("tf_cache_time", cache_time, cache_time);

            buffer = std::make_shared<tf2_ros::Buffer>(ros::Duration(cache_time));
            listener = std::make_unique<tf2_ros::TransformListener>(dynamic_cast<tf2_ros::Buffer &>(*buffer), nh, true);
            ROS_INFO("Using local tf buffer (cache %.3g s).", cache_time);
        }
    }
    return buffer;
}

} // namespace
