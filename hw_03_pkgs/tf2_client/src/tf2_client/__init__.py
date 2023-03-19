from __future__ import absolute_import, division, print_function
import rospy
from tf2_ros import Buffer, TransformListener, BufferClient
from threading import Lock

__all__ = [
    'clear',
    'get_buffer'
]

_lock = Lock()
_buffer = None
_listener = None


def get_buffer():
    global _buffer, _listener
    with _lock:
        if _buffer is None:
            server = rospy.get_param('~tf_server', None)
            if server:
                check_frequency = rospy.get_param('~tf_check_frequency', None)
                timeout_padding = rospy.get_param('~tf_timeout_padding', 2.0)
                _buffer = BufferClient(server, check_frequency, rospy.Duration.from_sec(timeout_padding))
                rospy.loginfo('Using tf buffer client (server %s, timeout padding %.3g s).',
                              server, timeout_padding)
            else:
                cache_time = rospy.get_param('~tf_cache_time', 10.0)
                queue_size = rospy.get_param('~tf_queue_size', None)
                buff_size = rospy.get_param('~tf_buff_size', 65536)
                _buffer = Buffer(rospy.Duration(cache_time))
                _listener = TransformListener(_buffer, queue_size, buff_size)
                rospy.loginfo('Using local tf buffer (cache %.3g s, queue size %s, buffer size %s).',
                              cache_time, queue_size, buff_size)

        return _buffer


def clear(buffer=True, listener=True):
    global _buffer, _listener
    with _lock:
        if listener and _listener is not None:
            _listener.unregister()
            _listener = None
            rospy.loginfo('Listener unregistered and deleted.')
        if buffer and _buffer is not None:
            _buffer = None
            rospy.loginfo('Buffer deleted.')
