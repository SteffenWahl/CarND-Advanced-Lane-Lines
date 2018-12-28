import numpy as np

# class RingBuffer copied from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html
class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.xdata = []
        self.ydata = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x, y):
            """ Append an element overwriting the oldest one. """
            self.xdata[self.cur] = x
            self.ydata[self.cur] = y
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.xdata[self.cur:]+self.xdata[:self.cur], self.ydata[self.cur:]+self.ydata[:self.cur]

    def append(self, x, y):
        """append an element at the end of the buffer"""
        self.xdata.append(x)
        self.ydata.append(y)
        if len(self.xdata) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.xdata, self.ydata 

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.buffer = RingBuffer(5)
        self.notdetected = 0

    def update(self, x, y):
        self.allx = x
        self.ally = y
        if len(x) < 50:
            self.detected = False
            self.notdetected = self.notdetected + 1
        else:
            self.detected = True
            self.current_fit = np.polyfit(y, x, 2)
            self.notdetected = 0
        self.buffer.append(x,y)
        ax, ay = self.buffer.get()
        aax = np.concatenate(ax)
        aay = np.concatenate(ay)
        self.best_fit = np.polyfit(aay,aax,2)

    def update_force(self, x, y):
        self.allx = x
        self.ally = y
        self.detected = True
        self.current_fit = np.polyfit(y, x, 2)
        self.notdetected = 0
        self.buffer.append(x,y)
        ax, ay = self.buffer.get()
        aax = np.concatenate(ax)
        aay = np.concatenate(ay)
        self.best_fit = np.polyfit(aay,aax,2)


    def y_range(self):
        ax, ay = self.buffer.get()
        aay = np.concatenate(ay)
        return (min(aay),max(aay))

    def get_radius(self):        
        fit = self.best_fit
        pos = max(self.ally)
        rad = (1+(2*fit[0]*pos + fit[1])**2)**(3/2)/abs(2*fit[0])
        return rad            

    def get_vals(self, rang=None):
        if rang is None:
            rang = self.y_range()
        ploty = np.linspace(rang[0], rang[1], rang[1]-rang[0] )
        fit = self.best_fit
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

        return fitx, ploty            

    def get_val(self, pos):
        fit = self.best_fit
        fitx = fit[0]*pos**2 + fit[1]*pos + fit[2]

        return fitx

