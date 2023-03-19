#include <mutex>
#include <ros/ros.h>
#include <tf2_ros/buffer_interface.h>

namespace tf2_client {

typedef std::shared_ptr<tf2_ros::BufferInterface> BufferPtr;

BufferPtr get_buffer(ros::NodeHandle &nh, ros::NodeHandle &pnh);

}
