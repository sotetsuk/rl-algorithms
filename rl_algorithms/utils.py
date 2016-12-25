class ExperiencePool(object):

    def __init__(self):
        self.memory = []

    def append(self, state, action, next_state, reward):
        pass

    def __iter__(self, n=None):
        for i, mem in enumerate(self.memory):
            if i >= n:
                break
            state, action, next_state, reward = mem
            yield state, action, next_state, reward