import tkinter as tk
import setproctitle
import random
import signal

import solver
import imrec

class GUI(object):
  def __init__(self, size):
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

  def update(self):
    screenshot = imrec.get_screenshot()
    sol = solver.solvestuff(screenshot)
    sol.solve()

  def signal(self, signal, idk):
    #print(signal, idk)
    #cols = ["black", "white", "blue", "magenta", "orange", "red"]
    #self.w.configure(background=random.choice(cols))
    self.update()

if __name__ == '__main__':
    setproctitle.setproctitle("PYSKLORT")
    GUI((1440,900))
