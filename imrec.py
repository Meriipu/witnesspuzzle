import itertools
import cv2
import numpy as np
import pyscreenshot

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

screenres = (1920, 1200)
gameres = (1440,900)
puzzlebox = [539,268,898,628]

def write_img(path, img, conversion=cv2.COLOR_RGB2BGR):
  if len(img.shape) > 2 and img.shape[2] > 1:
    if conversion:
      cv2.imwrite(path, cv2.cvtColor(img, conversion))
    else:
      cv2.imwrite(path, img)
  else:
    cv2.imwrite(path, img*255)

def get_screenshot(demo_img=None):
  startx,starty = (screenres[0] - gameres[0], 0)
  endx,endy = (startx+gameres[0], starty+gameres[1])

  if demo_img is None:
    img = np.array(pyscreenshot.grab())
  else:
    img = demo_img

  crop = img[starty:endy, startx:endx, :].copy()
  #write_img("./out/test_crop.png", crop)
  return crop

def imread(path):
  #img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
  #return img/255.0
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.uint8)
  return img

ring_template = imread('./templates/template.png')
knob_template = imread('./templates/16x_32y.png')

def template_matching(img, template, offset_xy=None, fn=None):
  # All the 6 methods for comparison in a list
  #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
  #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
  #method = cv2.TM_CCOEFF
  method = cv2.TM_CCOEFF_NORMED
  res = cv2.matchTemplate(img,template,method)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

  if offset_xy is None:
    h,w = np.shape(template)[:-1]
    pos = (max_loc[0] + w//2,  max_loc[1] + h//2)
  else:
    pos = (max_loc[0] + offset_xy[0], max_loc[1] + offset_xy[1])
  img = img.copy()
  img4 = cv2.circle(img, pos, 5, (125,0,125), 2)
  if fn is None:
    fn = "./out/test_template.png"
  #write_img(fn, img4)
  return (pos[1], pos[0])

class puzzlegen(object):
  def __init__(self, img):
    self.cellcounts = (4,4)
    #self.ideal_bg = (0, 143, 111)
    self.ideal_bg = (0,133,103)
    self.thresh = 60

    self.img = img
    self.ringpos = template_matching(self.img, ring_template, None, './out/test_ring_template.png')
    self.knobpos = template_matching(self.img, knob_template, (16,32), './out/test_knob_template.png')

    self.mono = self.cyanize(self.img)
    self.horstripes, self.verstripes = self.get_stripes()
    self.mids = self.get_cells()
    self.get_shapes()

    x0,y0,xn,yn = puzzlebox
    self.puzzimg = self.classimg[y0:yn,x0:xn,:].copy()
    write_img("./out/test_puzzimg.png", self.puzzimg)

  def cyanize(self, img):
    diff = np.sum(np.sqrt( (img - self.ideal_bg)**2), axis=2)
    diff = np.sqrt(np.sum( (img - self.ideal_bg)**2, axis=2))

    mono = np.where(diff < self.thresh, 1, 0)
    print(mono.shape)
    #write_img("./out/test_cyanize.png", mono)
    return mono

  def stripes_to_puzzle(self, src_vtx, dst_vtx):
    print(src_vtx, dst_vtx)
    verstripe1 = self.verstripes[src_vtx[1]]
    verstripe2 = self.verstripes[dst_vtx[1]]
    horstripe1 = self.horstripes[src_vtx[0]]
    horstripe2 = self.horstripes[dst_vtx[0]]

    x0,xn = verstripe1[1],verstripe2[1]
    y0,yn = horstripe1[0],horstripe2[0]
    xo,yo,_,_ = puzzlebox
    return (y0-yo,x0-xo),(yn-yo,xn-xo)

  def get_stripes(self):
    #x,y=572,597
    #y,x = self.ringpos
    hiy,lox = self.ringpos
    loy,hix = self.knobpos

    xlen = hix - lox
    xstep = xlen/self.cellcounts[1]
    self.xstep = xstep
    xposes = [int(np.round(lox + xstep*i)) for i in range(1,self.cellcounts[1])]
    xposes = [lox] + xposes + [hix]
    verstripes = [((loy, hiy), xpos) for xpos in xposes]

    ylen = hiy - loy
    ystep = ylen/self.cellcounts[0]
    self.ystep=ystep
    yposes = [int(np.round(loy + ystep*i)) for i in range(1,self.cellcounts[0])]
    yposes = [loy] + yposes + [hiy]
    horstripes =  [(ypos, (lox,hix)) for ypos in yposes]

    self.img2 = self.img.copy()
    for i,(ypos,(x0,xn)) in enumerate(horstripes):
      self.img2[ypos,  x0:xn] = (255, 0, 50*(i+1))
    for i,((y0,yn),xpos) in enumerate(verstripes):
      self.img2[y0:yn,xpos] = (50*(i+1),255,0)
    #write_img("./out/test_stripe.png", self.img2)

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
    #write_img("./out/test_cells.png", img3)
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

    write_img("./out/test_classes.png", img4)
    self.classimg = img4

    A = np.zeros(shape=self.cellcounts, dtype=np.uint8)
    mx = np.max(Q)
    for k,cls in enumerate(Q):
      if cls == bgcls:
        continue

      if cls == mx:
        plotcls = bgcls
      else:
        plotcls = cls
      i,j = backmap[k]
      A[i][j] = plotcls

    self.res = A
