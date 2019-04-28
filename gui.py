import tkinter as tk
import setproctitle
import random
import signal

import solver
import imrec
import time

class GUI(object):
  def __init__(self, size):
    self.size = size

    master = tk.Tk()
    master.geometry('%dx%d+%d+%d' % (size[0], size[1], 0, 0))

    master.bind("q", lambda *event: self.exit())

    self.master = master
    self.root = tk.Frame(master, width=size[0], height=size[1])
    self.root.propagate(False)
    self.root.grid()
    self.w = tk.Canvas(self.root, width=size[0], height=size[1])
    self.w.grid()

    signal.signal(signal.SIGUSR1, self.signal)

    self.master.after(1000, self.mcfungus)
    tk.mainloop()

  def mcfungus(self):
    self.master.update_idletasks()
    self.master.after(100, self.mcfungus)


  def exit(self):
    self.root.quit()

  def rgb2hex(self,rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

  def draw_board(self):
    self.tkimg = tk.PhotoImage(file='./out/test_puzzimg.png')
    self.canvas_tkimg = self.w.create_image(0,0, image=self.tkimg, anchor=tk.NW)
    self.master.update_idletasks()

  def draw_path(self, _visits, puz):
    v = _visits.copy()
    lines = []
    cur = v.pop(0)
    while v:
      self.master.update_idletasks()
      action = v.pop(0)
      if action == "undo":
        prev,coords,line = lines.pop()
        self.w.delete(line)
        cur = prev
        continue
      elif action == "solution":
        return
      else:
        (y0,x0),(yn,xn) = puz.stripes_to_puzzle(cur, action)
        line = self.w.create_line(x0,y0,xn,yn,width=4,fill="orange")
        coords = (y0,x0),(yn,xn)
        cur = action
        lines.append((cur,coords,line))
        continue

  def update(self):
    screenshot = imrec.get_screenshot()
    sol = solver.solvestuff(screenshot)
    self.draw_board()
    sol.solve()
    if sol.solutions:
      s = min(sol.solutions, key=len)
      self.draw_path(s, sol.puz)
    else:
      self.w.create_line(0,0,self.size[0],self.size[1],width=4,fill="red")
      self.w.create_line(0,self.size[1],self.size[0],0,width=4,fill="red")

  def signal(self, signal, idk):
    #print(signal, idk)
    #cols = ["black", "white", "blue", "magenta", "orange", "red"]
    #self.w.configure(background=random.choice(cols))
    self.update()

if __name__ == '__main__':
    setproctitle.setproctitle("PYSKLORT")
    x0,y0,xn,yn = imrec.puzzlebox
    GUI((xn-x0,yn-y0))
