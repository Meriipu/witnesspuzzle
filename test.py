
def fromto(old,new,h=5,w=5):
  y0,x0 = old
  yn,xn = new
  dy,dx = yn-y0, xn-x0

  if (y0 == yn == 0 or y0 == yn == h)  or (x0 == xn == 0 or x0 == xn == w):
    return None
  elif dx == 0:
    y = min(y0,yn)
    x1,x2 = x0-1,x0
    return (y,x1), (y,x2)
  elif dy == 0:
    x = min(x0,xn)
    y1,y2 = y0-1,y0
    return (y1,x), (y2,x)
  else:
    return None