import numpy as np
from libs.practical00_lib import circle, rectangle

x = np.array( [12,34,3.14159] )
print( x.T.shape )

y = np.array( [[12,34,31]] )
print( y.T.shape )

z = np.array( [12,1,42,3.14] )
a = np.array( [[1,2,3,4],[5,6,7,8],[9,10,11,12]] )

print( z )
print( a )
print( a[1,1] )
print( z[-2] )

print( a%2 )

print( a-z )

print( z[::-1] )

np.random.seed( 1 )
b = np.random.randint( 0, 10, size=(3,4) )
print( b )

print( a*b )
print( a@b.T )



d, c, a = circle( 4 )
print( d, c, a )
d = circle(4, pi=np.pi )
print( d )
print( d[2] )

obj= rectangle( 4,3 )
# obj()
obj.circ()
obj.area()
obj.name = 'thing'
print( f'This {obj.name} has a circ of {obj.c} and an area of {obj.a}' )