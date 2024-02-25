class MinMaxScaler:
    def __init__(self, min_val=0, max_val=480):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)

    def de_normalize(self, normalized_tensor):
        return (normalized_tensor * (self.max_val - self.min_val)) + self.min_val