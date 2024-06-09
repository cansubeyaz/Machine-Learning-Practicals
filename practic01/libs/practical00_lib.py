def circle( r, pi=3.14 ):
    d = 2*r
    c = 2*pi*r
    a = pi*r**2
    return d, c, a

class rectangle():
    def __init__(self, h, w ):
        self.h = h
        self.w = w
        if h == w:
            self.name = 'square'
        else:
            self.name = 'rectangle'



    def circ(self, ):
        self.c = 2*self.h+2*self.w

    def area(self ):
        self.a = self.h*self.w

    def __call__(self, ):
        self.circ()
        self.area()
        print( f'This {self.name} has a height of {self.h} and a width of {self.w}' )
        print( f'The circumference is {self.c} and the area is {self.a}' )