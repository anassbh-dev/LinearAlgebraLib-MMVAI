import time

class random:
    def __init__(self, *dimensions, seed=None):
        if seed is None:
            seed = int(time.time_ns())  # Use current time to seed the random number generator
        self.dimensions = dimensions
        self.seed = seed
        self.a = 156
        self.b = 108
        self.c = 847
        self.matrix = self.create_matrix(self.dimensions)

    def rand(self):
        self.seed = (self.a * self.seed + self.b) % self.c
        return self.seed % self.c

    def create_matrix(self, dims):
        if len(dims) == 1:
            return [self.rand() for _ in range(dims[0])]
        else:
            return [self.create_matrix(dims[1:]) for _ in range(dims[0])]

    def __getitem__(self, key):
        return self.matrix[key]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __iter__(self):
        return iter(self.matrix)

    def __str__(self):
        return str(self.matrix)

    def get_matrix(self):
        return self.matrix
