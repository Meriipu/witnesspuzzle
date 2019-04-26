import itertools
import cv2
import numpy as np

# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster



def write_img(path, img, conversion=cv2.COLOR_RGB2BGR):
  if len(img.shape) > 2 and img.shape[2] > 1:
    if conversion:
      cv2.imwrite(path, cv2.cvtColor(img, conversion))
    else:
      cv2.imwrite(path, img)
  else:
    cv2.imwrite(path, img*255)

def get_screenshot():
  screenres = (1920, 1200)
  gameres = (1440,900)
  startx,starty = (screenres[0] - gameres[0], 0)
  endx,endy = (startx+gameres[0], starty+gameres[1])

  img = get_demo_screenshot()

  crop = img[starty:endy, startx:endx, :]
  write_img("./test_crop.png", crop)
  return crop

def imread(path):
  #img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
  #return img/255.0
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.uint8)
  return img

def get_demo_screenshot():
  return imread(path)

ring_template = imread('./template.png')
knob_template = imread('./16x_32y.png')

def template_matching(img, template, offset_xy=None, fn=None):
  # All the 6 methods for comparison in a list
  #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
  #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
  #method = cv2.TM_CCOEFF
  method = cv2.TM_CCOEFF_NORMED
  res = cv2.matchTemplate(img,template,method)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

  #top_left = max_loc
  #bottom_right = (top_left[0] + w, top_left[1] + h)
  if offset_xy is None:
    h,w = np.shape(template)[:-1]
    pos = (max_loc[0] + w//2,  max_loc[1] + h//2)
  else:
    pos = (max_loc[0] + offset_xy[0], max_loc[1] + offset_xy[1])
  img = img.copy()
  img4 = cv2.circle(img, pos, 5, (125,0,125), 2)
  if fn is None:
    fn = "./test_template.png"
  write_img(fn, img4)
  return (pos[1], pos[0])

class puzzlegen(object):
  def __init__(self):
    self.cellcounts = (4,4)
    #self.ideal_bg = (0, 143, 111)
    self.ideal_bg = (0,133,103)
    self.thresh = 60

    self.img = get_screenshot()
    self.ringpos = template_matching(self.img, ring_template, None, 'test_ring_template.png')
    self.knobpos = template_matching(self.img, knob_template, (16,32), 'test_knob_template.png')

    self.mono = self.cyanize(self.img)
    self.horstripes, self.verstripes = self.get_stripes()
    self.mids = self.get_cells()
    self.get_shapes()

  def cyanize(self, img):
    diff = np.sum(np.sqrt( (img - self.ideal_bg)**2), axis=2)
    diff = np.sqrt(np.sum( (img - self.ideal_bg)**2, axis=2))

    mono = np.where(diff < self.thresh, 1, 0)
    print(mono.shape)
    write_img("./test_cyanize.png", mono)
    return mono

  def get_stripes(self):
    def longest_stretch(stripe, enabler_c=1, stretch_c=0):
      enabled = False

      longest = [None,None]
      attempt = [None,None]
      for i,x in enumerate(stripe):
        #if i%3 != 0:
        #  continue

        if x == enabler_c:
          enabled = True

        if not enabled:
          continue

        if x == stretch_c:
          if attempt[0] is None:
            attempt[0] = i
          attempt[1] = i
        else:
          if longest[0] is None and longest[1] is None:
            longest = attempt
          elif (attempt[0] is not None) and (attempt[1] is not None):
            if (longest[1] - longest[0]) < (attempt[1] - attempt[0]):
              longest = attempt
          attempt = [None,None]
      return longest

    #x,y=572,597
    y,x = self.ringpos
    xstripe = longest_stretch(self.mono[y,:])
    ystripe = longest_stretch(self.mono[:,x])

    #pad/adjust ends of stripe to avoid border
    #xstripe[1] -= 10
    #ystripe[0] += 10
    xstripe[0] = x
    ystripe[1] = y
    #cheap hack
    ystripe[0],xstripe[1] = self.knobpos

    xlen = xstripe[1] - xstripe[0]
    xstep = xlen/self.cellcounts[1]
    xposes = [int(np.round(xstripe[0] + xstep*i)) for i in range(1,self.cellcounts[1])]
    xposes = [xstripe[0]] + xposes + [xstripe[1]]
    verstripes = [((ystripe[0], ystripe[1]), xpos) for xpos in xposes]

    ylen = ystripe[1] - ystripe[0]
    ystep = ylen/self.cellcounts[0]
    yposes = [int(np.round(ystripe[0] + ystep*i)) for i in range(1,self.cellcounts[0])]
    yposes = [ystripe[0]] + yposes + [ystripe[1]]
    horstripes =  [(ypos, (xstripe[0], xstripe[1])) for ypos in yposes]

    self.img2 = self.img.copy()
    for i,(ypos,(x0,xn)) in enumerate(horstripes):
      self.img2[ypos,  x0:xn] = (255, 0, 50*(i+1))
    for i,((y0,yn),xpos) in enumerate(verstripes):
      self.img2[y0:yn,xpos] = (50*(i+1),255,0)
    write_img("./test_stripe.png", self.img2)

    return horstripes,verstripes

  def get_cells(self):
    A = np.array( [xpos for ((y0,yn),xpos) in self.verstripes],dtype=np.uint16 )
    xmid = ((A[:-1] + A[1:])//2)

    B = np.array( [ypos for (ypos,(x0,xn)) in self.horstripes],dtype=np.uint16 )
    ymid = ((B[:-1] + B[1:])//2)

    mids = list(itertools.product(enumerate(ymid),enumerate(xmid)))
    img3 = self.img2.copy()
    for (i,y),(j,x) in mids:
      img3 = cv2.circle(img3, (x,y), 5, (0,0,255), 2)
    write_img("./test_cells.png", img3)
    return mids

  def get_shapes(self):
    """find the colours of the cells"""
    colours = [[None for j in range(self.cellcounts[0])] for i in range(self.cellcounts[1])]
    X = []
    backmap = {}
    rad = 3
    for k,((i,y),(j,x)) in enumerate(self.mids):
      chunk = self.img[y-rad:y+rad, x-rad:x+rad,:]
      m = np.mean(np.mean(chunk, axis=0), axis=0).astype(np.uint16)
      colours[i][j] = m
      X.append(m)
      backmap[k] = (i,j)
    print(np.shape(X))
    Z = linkage(X, 'ward')
    Q = fcluster(Z, self.thresh, criterion='distance')

    closenesses = []
    for k,cls in enumerate(Q):
      i,j = backmap[k]
      closenesses.append( np.sqrt(np.sum( (colours[i][j] - self.ideal_bg)**2)   ) )
    minidx = np.argmin(closenesses)
    bgcls = Q[minidx]

    blibs = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
    img4 = self.img2.copy()
    for k,((i,y),(j,x)) in enumerate(self.mids):
      cls = Q[k]
      if cls == bgcls:
        continue
      col = blibs[(cls-1)]
      img4 = cv2.circle(img4, (x,y), 5, col, 2)

    write_img("./test_classes.png", img4)

if __name__ == '__main__':
  path = "./test.png"
  #path = "./test2.png"
  puzzlegen()