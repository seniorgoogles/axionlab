
class Sensitivity:
    
    def __init__(self, model, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis
        
        self.data_x_axis = list()
        self.data_y_axis = list()

    def add(self, x_data, y_data):
        self.data_x_axis.append(x_data)
        self.data_y_axis.append(y_data)

    def get_label_xaxis(self):
        return self.x_axis
    
    def get_label_yaxis(self):
        return self.y_axis
    
    def get_data(self):
        return self.data_x_axis, self.data_y_axis