import numpy as np
import imrec

def get_demo_screenshot():
  #path = "./testfiles/test3.png"
  path = "./testfiles/testB.png"
  return imrec.imread(path)


class solvestuff(object):
  def __init__(self, screenshot):
    self.puz = imrec.puzzlegen(screenshot)

    self.ch, self.cw = np.shape(self.puz.res)
    self.vh, self.vw = self.ch+1, self.cw+1
    self.vsize = self.vh, self.vw

    self.starting_vertex = (self.vh-1, 0)
    self.goal_vertex = (0, self.vw-1)
    self.goal_node = (0,self.cw-1)

  def neighbours(self,vtx,HW):
    i,j = vtx
    H,W = HW
    return {(y,x) for (y,x) in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)] if (0 <= y < H) and (0 <= x < W)}

  def check_neighbours(self, neighbour_listing, accept_goalmix, seeds=None):
    # if a wall was hit and a node is neighbours with a node of another
    # class, without also being a neighbour of the goal node, something
    # went wrong.
    def spancheck(node):
      okcls = self.puz.res[node[0]][node[1]]
      visited = set([node])
      stack = [node]
      mixed = False
      while stack:
        node = stack.pop()
        #if accept_goalmix and node == self.goal_node:
        #  return True, visited | {node}
        cls = self.puz.res[node[0]][node[1]]
        if okcls == 0:
          okcls = cls
        elif cls != okcls and cls != 0:
          mixed = True
        for nb in neighbour_listing[node]:
          if nb in visited:
            continue
          else:
            stack.append(nb)
            visited.add(nb)
      return not mixed, visited

    all_visits = set()
    if seeds:
      checklist = seeds
    else:
      checklist = set(neighbour_listing)

    for cell in checklist:
      if cell in all_visits:
        continue
      result,visited = spancheck(cell)
      if not result:
        if accept_goalmix and self.goal_node in visited:
          pass
        else:
          return False
      all_visits.update(visited)
    return True

  def fromto(self,old,new):
    """given two vertices, get the nodes that the line between them separates"""
    h,w = self.vsize
    y0,x0 = old
    yn,xn = new
    dy,dx = yn-y0, xn-x0

    if (y0 == yn == 0 or y0 == yn == h-1)  or (x0 == xn == 0 or x0 == xn == w-1):
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

  def solve(self):
    # mapping from cells to its connected neighbours
    # updated when connections are broken by edges
    #neighbour_listing = [[self.neighbours((i,j),(self.ch, self.cw)) for j in range(self.cw)] for i in range(self.ch)]
    neighbour_listing = {(i,j):self.neighbours((i,j),(self.ch, self.cw)) for j in range(self.cw) for i in range(self.ch)}

    solutions = []
    self.solutions = solutions

    derpstory = [self.starting_vertex]
    self.derpstory = derpstory

    visits = [self.starting_vertex]
    visit_set = set(visits)
    stack = [ self.neighbours(vtx=self.starting_vertex, HW=(self.vh,self.vw)) ]
    backtrack = False
    while stack:
      if visits[-1] == self.goal_vertex:
        if self.check_neighbours(neighbour_listing, accept_goalmix=False):
          print("woop")
          solutions.append(visits.copy())
          derpstory.append("solution")
          backtrack = True
        else:
          backtrack = True
      elif len(stack) == 0:
        print("empty stack")
        break

      # backtrack
      if backtrack or len(stack[-1]) == 0:
        backtrack = False
        derpstory.append("undo")
        delnode = visits.pop()
        visit_set.remove(delnode)
        stack.pop()
        if delnode == self.starting_vertex:
          break
        nownode = visits[-1]
        cells = self.fromto(delnode, nownode)
        if cells:
          c1,c2 = cells
          neighbour_listing[c1].add(c2)
          neighbour_listing[c2].add(c1)
        continue

      node = stack[-1].pop()

      derpstory.append(node)
      visits.append(node)
      visit_set.add(node)
      unvisited_neighbours = self.neighbours(node,self.vsize) - visit_set
      stack.append(unvisited_neighbours)

      prevnode = visits[-2]
      cells = self.fromto(prevnode, node)
      if cells:
        c1,c2 = cells
        neighbour_listing[c1].remove(c2)
        neighbour_listing[c2].remove(c1)
      # edgecheck
      edgecheck = lambda i,j: i == 0 or j == 0 or i == self.vh-1 or j == self.vw-1
      if node != self.starting_vertex and node != self.goal_vertex:
        if edgecheck(*node) and not edgecheck(*prevnode):
          if cells:
            if not self.check_neighbours(neighbour_listing, accept_goalmix=True, seeds=cells):
              backtrack = True
              continue
    print("\n\n")
    print(derpstory)
    print("\n")
    for solution in solutions:
      print(solution)

if __name__ == '__main__':
  solver = solvestuff(get_demo_screenshot())
  solver.solve()