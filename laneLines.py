# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import cv2
from warnings import warn


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    vs = vertices.shape
    if len(vs) == 2:
        vertices= vertices.reshape((1, vs[0], vs[1]))
    mask = np.zeros_like(img)   
    
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, xyxyVecs, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in xyxyVecs:
        x1, y1, x2, y2 = line.astype(int)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns hough lines as starting and ending points [x1, y1, x2, y2].
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        lines = np.empty((0, 1, 4))
    return lines


def draw_hough_lines(xyxyVecs, rows, cols, **kwargs):
    line_img = np.zeros((rows, cols, 3), dtype=np.uint8)
    draw_lines(line_img, xyxyVecs, **kwargs)
    return line_img


def fake_color(bw, color=[1., 0, 0]):
    r, g, b = color
    return np.dstack((r * bw, b * bw, g * bw)).astype(bw.dtype)


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(img, alpha, initial_img, beta, gamma)


def drawPolygon(vertices, ax=None, **kwargs):
    for (k, v) in dict(edgecolor='blue', lw=4, facecolor='none', linestyle='--').items():
        if k not in kwargs:
            kwargs[k] = v
    if ax is None:
        ax = plt.gca()
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    return ax.add_patch(Polygon(vertices, True, **kwargs))


class Pipeline(object):
    '''Pipeline for lane-line-finding.'''
    
    def __init__(self, horizon=.6, horizonRadius=None, hoodClearance=0):
        self.horizon = horizon
        self.horizonRadius = horizonRadius
        self.hoodClearance = hoodClearance

    @property
    def vertices(self):
        if self.horizonRadius is None:
            calibration = .04
            # Calibrate for a horizonRadius of `calibration` at horizon=.6
            x1 = (1 - self.horizon) * (.5 - calibration) / .4
            self.horizonRadius = .5 - x1
        x = self.cols
        y = self.rows
        return np.array([
                    (0, y * (1. - self.hoodClearance)), 
                    (x * (.5 - self.horizonRadius), y * self.horizon),
                    (x * (.5 + self.horizonRadius), y * self.horizon), 
                    (x, y * (1. - self.hoodClearance))
                   ], dtype=np.int32)

    def prepare(self, image):
        self.rows, self.cols = image.shape[:2]
        out = grayscale(image)
        out = gaussian_blur(out, 5)
        out = canny(out, 50, 150)
        out = region_of_interest(out, self.vertices)
        return out

    def findLines(self, preparedImage):
        lines = [LineSegment(l) for l in hough_lines(preparedImage, 20, np.pi / 120, 42, 50, 20)]
        return lines

    def deduplicate_lines(self, lines,  maxabsthresh=64):
        '''Remove near-duplicates from a collection of lines.'''
        # TODO: This could probably be replaced with something more standard like k-means.
        def mn(v):
            return sum(v) / len(v)
        nl = len(lines)
        lineset = []
        for lk in lines:
            lk = lk.xyxy
            if len(lineset) == 0:
                lineset.append([lk, [lk]])
            else:
                classIndex = False
                for i, (ljmean, ljclass) in enumerate(lineset):
                    e = max(abs(lk - ljmean).ravel())
                    if e < maxabsthresh:
                        classIndex = i
                if not classIndex:
                    lineset.append([lk, [lk]])
                else:
                    lineset[classIndex][1].append(lk)
                    lineset[classIndex][0] = mn(lineset[classIndex][1])
        deduplicatedLines = [LineSegment(cl[0]) for cl in lineset]
        vy = self.vertices[:, 1]
        for l in deduplicatedLines:
            l.extendToHorizontalBorders(top=max(vy), bottom=min(vy))
        return deduplicatedLines

    def bakeLines(self, originalImage, preparedImage, lines, thickness=12, color=(232, 119, 34)):
        return weighted_img(
            draw_hough_lines([l.xyxy for l in lines], self.rows, self.cols, thickness=thickness, color=color),
            originalImage
        )

    def __call__(self, image):
        preparedImage = self.prepare(np.copy(image))
        allLines = self.findLines(preparedImage)
        lines = self.deduplicate_lines(allLines)
        return self.bakeLines(image, preparedImage, lines)
        
    def prettyShow(self, image):
        '''Make a nicer annotated figure.'''
        fig, ax = plt.subplots()
        preparedImage = self.prepare(image)
        allLines = self.findLines(preparedImage)
        lines = self.deduplicate_lines(allLines)
        ax.imshow(image)
        for l in lines:
            l.plotline(ax, lw=8, linestyle='-', color='orange', alpha=.5)
        for l in allLines:
            l.plotline(ax, lw=1, linestyle='-', color='magenta', alpha=.5)
        v = self.vertices.squeeze()
        drawPolygon(v, ax, alpha=.1, edgecolor='black', facecolor='orange')
        ax.grid(True)
        ax.set_ylim((image.shape[0], 0))
        ax.set_xlim((0, image.shape[1]));
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def processVideo(self, inpath, outpath, audio=False, subsection=None, show=True):
        from moviepy.editor import VideoFileClip
        inclip = VideoFileClip(inpath)
        if subsection is not None:
            inclip = inclip.subclip(*subsection)
        outclip = inclip.fl_image(self)
        outclip.write_videofile(outpath, audio=audio)
        if show:
            from IPython.display import HTML
            return HTML("""
                <video width="960" height="540" controls loop autoplay>
                  <source src="{0}">
                </video>
            """.format(outpath))


def saveImage(data, path):
    from PIL import Image
    Image.fromarray(
        (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    ).save(path)


class LineSegment(object):
    '''Math and plotting methods for lines.'''

    def __init__(self, xyxy):

        # Ensure line segment starts with its lower point.
        xyxy = np.asarray(xyxy).ravel()
        x1, y1, x2, y2 = xyxy
        if y1 > y2:
            xyxy = np.array([x2, y2, x1, y1]).reshape(np.asarray(xyxy).shape)

        self.xyxy = xyxy
        self.m = (y2 - y1) / (x2 - x1) 
        self.b = y1 - self.m * x1

    def y(self, x):
        return self.m * x + b

    def x(self, y):
        return (y - self.b) / self.m

    def extendToHorizontalBorders(self, top=None, bottom=0):
        if self.m == 0:
            return # Can't extend up or down!
        if top is None:
            top = self.rows
        if self.xyxy[1] > bottom:
            self.xyxy[0] = self.x(bottom)
            self.xyxy[1] = bottom
        if self.xyxy[3] < top:
            self.xyxy[2] = self.x(top)
            self.xyxy[3] = top

    def plotline(self, ax, extraxy=None, **kwargs):
        '''Pretty-plot a line on an axis.'''
        l = self.xyxy
        x = [l[0], l[2]]
        y = [l[1], l[3]]
        if extraxy is not None:
            x.append(extraxy[0])
            y.append(extraxy[1])
        return ax.plot(x, y, **kwargs)


def bk():
    from IPython.core.debugger import set_trace
    set_trace()
