import numpy as np
import imrec

def get_demo_screenshot():
  #path = "./testfiles/test.png"
  path = "./testfiles/test2.png"
  return imrec.imread(path)


class solvestuff(object):
  def __init__(self):
    screenshot = imrec.get_screenshot(get_demo_screenshot())
    self.puz = imrec.puzzlegen(screenshot)

    self.ch, self.cw = np.shape(self.puz.res)
    self.cells = [[self.neighbours((i,j),(self.ch, self.cw)) for j in range(self.cw)] for i in range(self.ch)]
    self.vh, self.vw = self.ch+1, self.cw+1
    self.starting_vertex = (self.vh-1, 0)

  def neighbours(self,vtx,HW):
    i,j = vtx
    H,W = HW
    return [(y,x) for (y,x) in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)] if (0 <= y < H) and (0 <= x < W)]

  def solve(self):
    visits = [self.starting_vertex]
    stack = [self.neighbours(self.starting_vertex, (self.vh,self.vw))]
    while stack:
      break

if __name__ == '__main__':
  solver = solvestuff()
  solver.solve()