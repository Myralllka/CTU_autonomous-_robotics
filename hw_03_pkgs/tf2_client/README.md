# tf2_client
Configurable tf2 client wrapper switching between local and remote buffer.

## Usage
```python
rospy.init_node('awesome_node')
buffer = tf2_client.get_buffer()
buffer.lookup_transform(target_frame, source_frame, time)
```

## Parameters 
- For remote buffer (parameters of `tf2_ros.BufferClient`):
    - `~tf_server`, defaults to `None`
    - `~tf_check_frequency`, `None`
    - `~tf_timeout_padding`, `2.0`
- For local buffer (parameters of `tf2_ros.Buffer` and `tf2_ros.TransformListener`):
    - `~tf_cache_time`, `10.0`
    - `~tf_queue_size`, `None`
    - `~tf_buff_size`, `65536`

Remote buffer is used if parameter `~tf_server` is specified.
