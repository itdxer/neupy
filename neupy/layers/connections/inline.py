LEFT = 'left'
RIGHT = 'right'
AND = 'bool'


class GlobalConnectionState(dict):
    def key_to_id(self, key):
        return "[{}] {!r}".format(id(key), key)

    def __setitem__(self, key, value):
        key_id = self.key_to_id(key)
        return super(GlobalConnectionState, self).__setitem__(key_id, value)

    def __getitem__(self, key):
        key_id = self.key_to_id(key)
        return super(GlobalConnectionState, self).__getitem__(key_id)

    def __contains__(self, key):
        key_id = self.key_to_id(key)
        return super(GlobalConnectionState, self).__contains__(key_id)


class EventTracker(GlobalConnectionState):
    def __getitem__(self, key):
        if key not in self:
            self[key] = []
        return super(EventTracker, self).__getitem__(key)

    def add(self, key, event):
        self[key].append(event)

    def activate_on(self, key, events):
        n_events = len(events)
        last_n_events = self[key][-n_events:]
        return last_n_events == events

    def is_gt_before_gt(self, key):
        return self.activate_on(key, [RIGHT, AND, LEFT])

    def is_lt_before_lt(self, key):
        return self.activate_on(key, [LEFT, AND, RIGHT])


class InlineConnection(object):
    left_states = GlobalConnectionState()
    right_states = GlobalConnectionState()
    events = EventTracker()

    def compare(self, left, right):
        self.events.add(left, LEFT)
        self.events.add(right, RIGHT)

        main_left, main_right = left, right

        if self.events.is_gt_before_gt(left):
            left = self.right_states[left]

        if self.events.is_lt_before_lt(right):
            right = self.left_states[right]

        connection = self.connect(left, right)

        self.left_states[main_left] = connection
        self.right_states[main_right] = connection

        return connection

    def __gt__(self, other):
        return self.compare(self, other)

    def __lt__(self, other):
        return self.compare(other, self)

    def __nonzero__(self):
        # For python 2 compatibility
        return self.__bool__()

    def __bool__(self):
        left_raw = self.left_raw
        right_raw = self.right_raw

        self.events.add(left_raw, AND)
        self.events.add(right_raw, AND)

        return True
