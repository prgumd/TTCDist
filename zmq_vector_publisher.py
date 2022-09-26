###############################################################################
#
# File: zmq_vector_publisher.py
# Available under MIT license
#
# An object that publishes timestamped floating point vectors over a zmq socket
#
# History:
# 09-01-21 - Levi Burner - Created file
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import zmq
import struct

class ZMQVectorPublisher(object):
    def __init__(self, port=5556, base_topic='ttc_depth'):
        self._port = port
        self._base_topic = base_topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind('tcp://*:{}'.format(self._port))

    def publish(self, topic, t, x, flags=0, copy=True, track=False):
        full_topic = self._base_topic + '/' + topic
        self._socket.send(full_topic.encode('ascii'), flags|zmq.SNDMORE)

        self._socket.send(struct.pack('d', float(t)), flags|zmq.SNDMORE)

        md = dict(
            dtype = str(x.dtype),
            shape = x.shape,
        )

        self._socket.send_json(md, flags|zmq.SNDMORE)
        self._socket.send(x, flags, copy=copy, track=track)


class ZMQVectorPublisherSaver(ZMQVectorPublisher):
    def __init__(self, port=5556, base_topic='ttc_depth'):
        super().__init__(port=port, base_topic=base_topic)
        self._save_dict = {}

    def publish(self, topic, t, x, flags=0, copy=True, track=False):
        if topic not in self._save_dict:
            self._save_dict[topic] = []

        self._save_dict[topic].append((t, x))

        super().publish(topic, t, x, flags=flags, copy=copy, track=track)

    def get_data(self):
        return self._save_dict
