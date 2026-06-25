import time
import queue
import threading
from typing import Dict, List, Any, Optional

class Message:
    def __init__(self, topic: str, value: Any, key: Optional[str] = None, partition: int = 0, offset: int = 0):
        self.topic = topic
        self.value = value
        self.key = key
        self.partition = partition
        self.offset = offset
        self.timestamp = time.time()

    def __repr__(self):
        return f"Message(topic={self.topic}, offset={self.offset}, key={self.key}, val_type={type(self.value).__name__})"


class MockKafkaBroker:
    """
    A thread-safe, in-memory message broker simulating Apache Kafka topics and partitions.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MockKafkaBroker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._topics: Dict[str, List[Message]] = {}
        self._topic_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._initialized = True

    def create_topic(self, topic: str):
        with self._global_lock:
            if topic not in self._topics:
                self._topics[topic] = []
                self._topic_locks[topic] = threading.Lock()

    def send(self, topic: str, value: Any, key: Optional[str] = None) -> Message:
        if topic not in self._topics:
            self.create_topic(topic)
        
        with self._topic_locks[topic]:
            offset = len(self._topics[topic])
            msg = Message(topic=topic, value=value, key=key, partition=0, offset=offset)
            self._topics[topic].append(msg)
            return msg

    def get_messages(self, topic: str, start_offset: int, limit: int = 100) -> List[Message]:
        if topic not in self._topics:
            return []
        
        with self._topic_locks[topic]:
            topic_msgs = self._topics[topic]
            if start_offset >= len(topic_msgs):
                return []
            end = min(start_offset + limit, len(topic_msgs))
            return topic_msgs[start_offset:end]

    def get_latest_offset(self, topic: str) -> int:
        if topic not in self._topics:
            return 0
        with self._topic_locks[topic]:
            return len(self._topics[topic])

    def clear(self):
        with self._global_lock:
            self._topics.clear()
            self._topic_locks.clear()


class Producer:
    def __init__(self, client_id: str = "default-producer"):
        self.client_id = client_id
        self.broker = MockKafkaBroker()

    def send(self, topic: str, value: Any, key: Optional[str] = None) -> Message:
        return self.broker.send(topic, value, key)


class Consumer:
    def __init__(self, topic: str, group_id: str = "default-group", auto_offset_reset: str = "earliest"):
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.broker = MockKafkaBroker()
        self.broker.create_topic(topic)
        
        # Track offsets locally per consumer group/topic
        self.current_offset = 0
        if auto_offset_reset == "latest":
            self.current_offset = self.broker.get_latest_offset(topic)

    def poll(self, timeout_ms: int = 100, max_records: int = 1) -> List[Message]:
        """
        Polls for new messages. Mimics Kafka consumer poll.
        """
        # Sleep for a tiny amount of time to simulate network polling
        time.sleep(min(timeout_ms / 1000.0, 0.05))
        
        msgs = self.broker.get_messages(self.topic, self.current_offset, limit=max_records)
        if msgs:
            self.current_offset = msgs[-1].offset + 1
        return msgs

    def commit(self):
        # In this mock, committing simply increments our tracking, which we do automatically on poll.
        pass

    def seek_to_beginning(self):
        self.current_offset = 0

    def seek_to_end(self):
        self.current_offset = self.broker.get_latest_offset(self.topic)
