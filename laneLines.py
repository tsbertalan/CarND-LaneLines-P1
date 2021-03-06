# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import cv2
from warnings import warn

from collections import deque
from sklearn.cluster import KMeans as ClusterFinder


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

    Color is RGB.
    """
    for line in xyxyVecs:
        x1, y1, x2, y2 = line.astype(int)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            
def hough_lines(img, rho, theta, threshold, minLineLen, maxLineGap):
    """
    `img` should be the output of a Canny transform.
        
    Returns hough lines as starting and ending points [x1, y1, x2, y2].
    """
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]), 
        minLineLength=minLineLen, maxLineGap=maxLineGap
        )
    if lines is None:
        lines = np.empty((0, 1, 4))
    return lines


def draw_hough_lines(xyxyVecs, rows, cols, **kwargs):
    line_img = np.zeros((rows, cols, 3), dtype=np.uint8)
    draw_lines(line_img, xyxyVecs, **kwargs)
    return line_img


def fake_color(bw, color=[1., 1., 1.]):
    r, g, b = color
    return np.dstack((r * bw, b * bw, g * bw)).astype(bw.dtype)


def to_color_if_bw(bw, color=[1., 1., 1.]):
    if len(bw.shape) == 2 or bw.shape[2] == 1:
        bw = fake_color(bw)
    return bw


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * beta + img * alpha + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(to_color_if_bw(initial_img), beta, to_color_if_bw(img), alpha, gamma)


def draw_polygon(vertices, ax=None, **kwargs):
    for (k, v) in dict(edgecolor='blue', lw=4, facecolor='none', linestyle='--').items():
        if k not in kwargs:
            kwargs[k] = v
    if ax is None:
        ax = plt.gca()
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    return ax.add_patch(Polygon(vertices, True, **kwargs))


def pixelwise_max(*imgs):
    return np.stack(imgs).max(axis=0)


class Pipeline(object):
    '''Pipeline for lane-line-finding.'''
    
    def __init__(self, 
        horizon=.6, horizonRadius=None, hoodClearance=0,
        gaussianRadius=5,
        lowCannyThreshold=50, highCannyThreshold=150,
        rho=20, theta=np.pi/120, houghThreshold=64, minLineLen=50, maxLineGap=20,
        minAbsSlope=.5, maxAbsSlope=2,
        ):
        '''
        Parameters
        ----------
        horizon : float, optional
            How far down on the image the top edge of the search region trapezoid
            should be. Specified as a number in (0.0, 1.0),
            where 0.0 is at the top of the image and 1.0, the bottom.
        hozironRadius : float, optional
            Half-width of the top edge of the search region trapezoid. in units of (sub)pixels.
        gaussianRadius : int, optional
            Diameter of Gaussian pre-smoothing.
        lowCannyThreshold : int, optional
        highCannyThreshold : int, optional
            Hysteresis thresholds for Canny edge detection.
        rho : float, optional
            Radius increment for assembling the Hough accumulator array. In units of pixels.
        theta : float, optional
            Angle increment for assembling the Hough accumulator array. In units of radians.
        houghThreshold : int, optional
            Minimum number of accumulated Hough votes for inclusion.
        minLineLeng : float, optional
            "Minimum length of line. Line segments shorter than this are rejected."
        maxLineGap : float, optional
            "Maximum allowed gap between line segments to treat them as single line."
        minAbsSlope : float, optional
            Lower threshold on line slope for filtering Hough candidates
        maxAbsSlope : float, optional
            Upper threshold on line slope for filtering Hough candidates
        '''

        self.horizon = horizon
        self.horizonRadius = horizonRadius
        self.hoodClearance = hoodClearance

        self.gaussianRadius = gaussianRadius
        self.lowCannyThreshold = lowCannyThreshold
        self.highCannyThreshold = highCannyThreshold

        self.rho = rho
        self.theta = theta
        self.houghThreshold = houghThreshold
        self.minLineLen = minLineLen
        self.maxLineGap = maxLineGap

        self.minAbsSlope = minAbsSlope
        self.maxAbsSlope = maxAbsSlope

        self.debug = False
        self.debugThickness = 4
        self.debugColor = [0, 255, 0]

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
        rgb = []
        for k in range(3):
            out = image[:, :, k]
            out = gaussian_blur(out, self.gaussianRadius)
            out = canny(out, self.lowCannyThreshold, self.highCannyThreshold)
            rgb.append(out)
        out = pixelwise_max(*rgb)
        out = region_of_interest(out, self.vertices)
        return out

    def find_lines(self, preparedImage):
        return [
            LineSegment(l)
            for l in hough_lines(
                preparedImage, self.rho, self.theta, self.houghThreshold,
                self.minLineLen, self.maxLineGap
            )
        ]

    def check_line(self, l):
        return (
            np.abs(l.m) >= self.minAbsSlope
            and np.abs(l.m) <= self.maxAbsSlope 
            and np.abs(l.m) != np.inf
        )

    def deduplicate_lines(self, lines):
        '''Remove near-duplicates from a collection of lines.'''
        if len(lines) > 1:
            
            points = [
                (l.m, l.b)
                for l in lines
                if self.check_line(l)
            ]
            if len(points) <= 1:
                return self.deduplicate_lines([LineSegment(mb=pt) for pt in points])
            clusterFinder = ClusterFinder(n_clusters=2, n_jobs=1)
            clusterFinder.fit(np.vstack(points))
            deduplicatedLines = [
                LineSegment(xyxy=None, mb=mb)
                for mb in clusterFinder.cluster_centers_
            ]
        else:
            deduplicatedLines = [l.copy() for l in lines]

        # Ensure that the up-sloping line is always second.
        if len(deduplicatedLines) == 2:
            if deduplicatedLines[0].m > deduplicatedLines[1].m:
                deduplicatedLines = deduplicatedLines[::-1]

        return deduplicatedLines

    def smooth_history(self, lines, nsmooth):
        if not hasattr(self, 'lineHistory'):
            
            self.lineHistory = deque(maxlen=nsmooth)
        if len(lines) == 2:
            self.lineHistory.append(lines)
        mbs = [
            [
                (linePair[k].m, linePair[k].b)
                for linePair in self.lineHistory
            ]
            for k in range(2)
        ]
        ms = [np.mean([mb[0] for mb in mbsk]) for mbsk in mbs]
        bs = [np.mean([mb[1] for mb in mbsk]) for mbsk in mbs]
        lines = [
            LineSegment(mb=(m, b))
            for (m, b) in zip(ms, bs)
        ]
        return lines

    def bake_lines(self, originalImage, lines, thickness=12, color=(232, 119, 34), alpha=1):
        return weighted_img(
            draw_hough_lines([l.xyxy for l in lines], self.rows, self.cols, thickness=thickness, color=color),
            originalImage,
            alpha=alpha
        )

    def __call__(self, image, smoothing=False, nsmooth=12):
        # Preprocess the image.
        preparedImage = self.prepare(np.copy(image))

        # Find the lines.
        allLines = self.find_lines(preparedImage)
        lines = self.deduplicate_lines(allLines)
        if smoothing:
            lines = self.smooth_history(lines, nsmooth)
        self.recut_lines(lines)

        # Draw the lines.
        if self.debug:
            baked = np.copy(preparedImage)
            filteredLines = [
                l for l in allLines
                if self.check_line(l)
            ]
            baked = self.bake_lines(baked, filteredLines, color=self.debugColor, thickness=self.debugThickness)

            # Put some info in the top left corner.
            infoImg = pad_image(self.make_mb_distribution_plot_image(filteredLines, lines)[:, :, :3], baked.shape)
            baked = weighted_img(infoImg, baked, alpha=1)

        else:
            baked = np.copy(image)
        baked = self.bake_lines(baked, lines, alpha=.5)
        return baked

    def recut_lines(self, lines):
        vy = self.vertices[:, 1]
        for l in lines:
            l.extend_to_horizontal_borders(top=max(vy), bottom=min(vy))
        return lines
        
    def pretty_show(self, image):
        '''Make a nicer annotated figure.'''
        fig, ax = plt.subplots()
        preparedImage = self.prepare(image)
        allLines = self.find_lines(preparedImage)
        lines = self.recut_lines(self.deduplicate_lines(allLines))
        ax.imshow(image)
        for l in lines:
            l.plot_line(ax, lw=8, linestyle='-', color='orange', alpha=.9)
        for l in allLines:
            l.plot_line(ax, lw=1, linestyle='-', color='magenta', alpha=.5)
        v = self.vertices.squeeze()
        draw_polygon(v, ax, alpha=.1, edgecolor='black', facecolor='orange')
        ax.grid(True)
        ax.set_ylim((image.shape[0], 0))
        ax.set_xlim((0, image.shape[1]));
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax
        
    def process_video(self, inpath, outpath, audio=False, subsection=None, show=True, smoothing=True, nsmooth=12):
        from moviepy.editor import VideoFileClip
        inclip = VideoFileClip(inpath)
        if subsection is not None:
            inclip = inclip.subclip(*subsection)
        outclip = inclip.fl_image(lambda img: self(img, smoothing=smoothing, nsmooth=nsmooth))
        outclip.write_videofile(outpath, audio=audio)
        if show:
            from IPython.display import HTML
            return HTML("""
                <video width="960" height="540" controls loop autoplay>
                  <source src="{0}">
                </video>
            """.format(outpath))

    def make_mb_distribution_plot_image(self, lines, linesdedup):
        '''Make a plot with some extra information to put in the corner of the video.'''
        import matplotlib as mpl
        for k in 'xtick', 'ytick', 'axes':
            mpl.rc(k, labelsize=10)
        l2p = lambda ls: np.stack([
                (l.m, l.b)
                for l in ls
                if self.check_line(l)
            ])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(l2p(lines)[:, 0], l2p(lines)[:, 1], color='yellow', s=24)
        ax.scatter(l2p(linesdedup)[:, 0], l2p(linesdedup)[:, 1], color='red', s=128, marker='*')
        ax.set_xlabel('slope $m$', color='white'); ax.set_ylabel('intercept $b$', color='white')

        ax.set_xlim(-self.maxAbsSlope, self.maxAbsSlope)
        ax.set_ylim(-2000, 2000)
        ax.patch.set_facecolor('black')
        fig.patch.set_facecolor('black')

        fig.tight_layout()
        out = fig2data(fig)
        plt.close(fig)
        return out


def pad_image(small, targetShape, padValue=0):
    if len(targetShape) == 2:
        targetShape = list(targetShape)
        targetShape.append(3)
    out = np.ones(targetShape, dtype=small.dtype) * padValue
    out[:small.shape[0], :small.shape[1], :] = to_color_if_bw(small)
    return out


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ()
 
    # Get the RGBA buffer from the figure
    h, w = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def save_image(data, path):
    from PIL import Image
    Image.fromarray(
        (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    ).save(path)


class LineSegment(object):
    '''Math and plotting methods for lines.'''

    def __init__(self, xyxy=None, mb=None):

        if xyxy is not None:
            self.xyxy = np.asarray(xyxy).ravel()
            x1, y1, x2, y2 = self.xyxy
            self.m = (y2 - y1) / (x2 - x1) 
            self.b = y1 - self.m * x1
        elif mb is not None:
            m, b = mb
            assert m != np.inf, 'TODO: Need to do an r-theta parameterization.'
            self.m = m
            self.b = b
            x1 = 1
            x2 = 2
            y1 = self.y(x1)
            y2 = self.y(x2)
            self.xyxy = np.array([x1, y1, x2, y2])
        else:
            bk()
            raise ValueError('Need to pass xyxy or mb')
        self._lineup()

    def copy(self):
        return LineSegment(self.xyxy)

    def _lineup(self):
        '''Ensure line segment starts with its lower point.'''
        xyxy = np.asarray(self.xyxy).ravel()
        x1, y1, x2, y2 = xyxy
        if y1 > y2:
            xyxy = np.array([x2, y2, x1, y1]).reshape(np.asarray(xyxy).shape)
        self.xyxy = xyxy

    def y(self, x):
        return self.m * x + self.b

    def x(self, y):
        return (y - self.b) / self.m

    def extend_to_horizontal_borders(self, top, bottom):
        if self.m == 0:
            return # Can't extend up or down!
        self.xyxy[0] = self.x(bottom)
        self.xyxy[1] = bottom
        self.xyxy[2] = self.x(top)
        self.xyxy[3] = top

    def plot_line(self, ax, extraxy=None, **kwargs):
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
