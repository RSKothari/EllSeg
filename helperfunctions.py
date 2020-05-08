import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

from itertools import chain
from scipy.ndimage import distance_transform_edt as distance

EPS = 1e-40

# Helper classes
class my_ellipse():
    def __init__(self, param):
        '''
        Accepts parameterized form
        '''
        self.EPS = 1e-3
        if param is not list:
            self.param = param
            self.mat = self.param2mat(self.param)
            self.quad = self.mat2quad(self.mat)
            #self.Phi = self.recover_Phi()

    def param2mat(self, param):
        cx, cy, a, b, theta = tuple(param)
        H_rot = rotation_2d(-theta)
        H_trans = trans_2d(-cx, -cy)

        A, B = 1/a**2, 1/b**2
        Q = np.array([[A, 0, 0], [0, B, 0], [0, 0, -1]])
        mat = H_trans.T @ H_rot.T @ Q @ H_rot @ H_trans
        return mat

    def mat2quad(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.EPS, 'Conic form incorrect'
        a, b, c, d, e, f = mat[0,0], 2*mat[0, 1], mat[1,1], 2*mat[0, 2], 2*mat[1, 2], mat[-1, -1]
        return np.array([a, b, c, d, e, f])

    def quad2param(self, quad):
        mat = self.quad2mat(quad)
        param = self.mat2param(mat)
        return param

    def quad2mat(self, quad):
        a, b, c, d, e, f = tuple(quad)
        mat = np.array([[a, b/2, d/2], [b/2, c, e/2], [d/2, e/2, f]])
        return mat

    def mat2param(self, mat):
        assert np.sum(np.abs(mat.T - mat)) <= self.EPS, 'Conic form incorrect'
        # Estimate rotation
        theta = self.recover_theta(mat)
        # Estimate translation
        tx, ty = self.recover_C(mat)
        # Invert translation and rotation
        H_rot = rotation_2d(theta)
        H_trans = trans_2d(tx, ty)
        mat_norm = H_rot.T @ H_trans.T @ mat @ H_trans @ H_rot
        major_axis = np.sqrt(1/mat_norm[0,0])
        minor_axis = np.sqrt(1/mat_norm[1,1])
        area = np.pi*major_axis*minor_axis
        return np.array([tx, ty, major_axis, minor_axis, theta, area])

    def phi2param(self, xm, ym):
        '''
        Given phi values, compute ellipse parameters

        Parameters
        ----------
        Phi : np.array [5, ]
            for information on Phi values, please refer to ElliFit.
        xm : int
        ym : int

        Returns
        -------
        param : np.array [5, ].
            Ellipse parameters, [cx, cy, a, b, theta]

        '''
        try:
            x0=(self.Phi[2]-self.Phi[3]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            y0=(self.Phi[0]*self.Phi[3]-self.Phi[2]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            term2=np.sqrt(((1-self.Phi[0])**2+4*(self.Phi[1])**2))
            term3=(self.Phi[4]+(y0)**2+(x0**2)*self.Phi[0]+2*self.Phi[1])
            term1=1+self.Phi[0]
            print(term1, term2, term3)
            b=(np.sqrt(2*term3/(term1+term2)))
            a=(np.sqrt(2*term3/(term1-term2)))
            alpha=0.5*np.arctan2(2*self.Phi[1],1-self.Phi[0])
            model = [x0+xm, y0+ym, a, b, -alpha]
        except:
            print('Inappropriate model generated')
            model = [np.nan, np.nan, np.nan, np.nan, np.nan]
        if np.all(np.isreal(model)) and np.all(~np.isnan(model)) and np.all(~np.isinf(model)):
            model = model
        else:
            model = [-1, -1, -1, -1, -1]
        return model

    def recover_theta(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        #print('a: {}. b: {}. c: {}'.format(a, b, c))
        if abs(b)<=EPS and a<=c:
            theta = 0.0
        elif abs(b)<=EPS and a>c:
            theta=np.pi/2
        elif abs(b)>EPS and a<=c:
            theta=0.5*np.arctan2(b, (a-c))
        elif abs(b)>EPS and a>c:
            #theta = 0.5*(np.pi + np.arctan(b/(a-c)))
            theta = 0.5*np.arctan2(b, (a-c))
        else:
            print('Unknown condition')
        return theta

    def recover_C(self, mat):
        a, b, c, d, e, f = tuple(self.mat2quad(mat))
        tx = (2*c*d - b*e)/(b**2 - 4*a*c)
        ty = (2*a*e - b*d)/(b**2 - 4*a*c)
        return (tx, ty)

    def transform(self, H):
        '''
        Given a transformation matrix H, modify the ellipse
        '''
        mat_trans = np.linalg.inv(H.T) @ self.mat @ np.linalg.inv(H)
        return self.mat2param(mat_trans), self.mat2quad(mat_trans), mat_trans

    def recover_Phi(self):
        '''
        Generate Phi
        '''
        x, y = self.generatePoints(50, 'random')
        data_pts = np.stack([x, y], axis=1)
        ellipseFit = ElliFit(**{'data':data_pts})
        return ellipseFit.Phi

    def verify(self, pts):
        '''
        Given an array of points Nx2, verify the ellipse model
        '''
        N = pts.shape[0]
        pts = np.concatenate([pts, np.ones((N, 1))], axis=1)
        err = 0.0
        for i in range(0, N):
            err+=pts[i, :]@self.mat@pts[i, :].T # Note that the transpose here is irrelevant
        return np.inf if (N==0) else err/N

    def generatePoints(self, N, mode):
        '''
        Generates 8 points along the periphery of an ellipse. The mode dictates
        the uniformity between points.
        mode: str
        'equiAngle' - Points along the periphery with angles [0:45:360)
        'equiSlope' - Points along the periphery with tangential slopes [-1:0.5:1)
        'random' - Generate N points randomly across the ellipse
        '''

        a = self.param[2]
        b = self.param[3]

        alpha = (a*np.sin(self.param[-1]))**2 + (b*np.cos(self.param[-1]))**2
        beta = (a*np.cos(self.param[-1]))**2 + (b*np.sin(self.param[-1]))**2
        gamma = (a**2 - b**2)*np.sin(2*self.param[-1])

        if mode == 'equiSlope':
            slope_list = [1e-6, 1, 1000, -1]
            K_fun = lambda m_i:  (m_i*gamma + 2*alpha)/(2*beta*m_i + gamma)

            x_2 = [((a*b)**2)/(alpha + beta*K_fun(m)**2 - gamma*K_fun(m)) for m in slope_list]

            x = [(+np.sqrt(val), -np.sqrt(val)) for val in x_2]
            y = []
            for i, m in enumerate(slope_list):
                y1 = -x[i][0]*K_fun(m)
                y2 = -x[i][1]*K_fun(m)
                y.append((y1, y2))
            y_r = np.array(list(chain(*y))) + self.param[1]
            x_r = np.array(list(chain(*x))) + self.param[0]

        if mode == 'equiAngle':

            T = 0.5*np.pi*np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            N = len(T)
            x = self.param[2]*np.cos(T)
            y = self.param[3]*np.sin(T)
            H_rot = rotation_2d(self.param[-1])
            X1 = H_rot.dot(np.stack([x, y, np.ones(N, )], axis=0))

            x_r = X1[0, :] + self.param[0]
            y_r = X1[1, :] + self.param[1]

        elif mode == 'random':
            T = 2*np.pi*(np.random.rand(N, ) - 0.5)
            x = self.param[2]*np.cos(T)
            y = self.param[3]*np.sin(T)
            H_rot = rotation_2d(self.param[-1])
            X1 = H_rot.dot(np.stack([x, y, np.ones(N, )], axis=0))
            x_r = X1[0, :] + self.param[0]
            y_r = X1[1, :] + self.param[1]

        else:
            print('Mode is not defined')

        return x_r, y_r

class ElliFit():
    def __init__(self, **kwargs):
        self.data = np.array([]) # Nx2
        self.W = np.array([])
        self.Phi = []
        self.pts_lim = 6*2
        for k, v in kwargs.items():
            setattr(self, k, v)
        if np.size(self.W):
            self.weighted = True
        else:
            self.weighted = False
        if np.size(self.data) > self.pts_lim:
            self.model = self.fit()
            self.error = np.mean(self.fit_error(self.data))
        else:
            self.model = [-1, -1, -1, -1, -1]
            self.Phi = [-1, -1, -1, -1, -1]
            self.error = np.inf

    def fit(self):
        # Code implemented from the paper ElliFit
        xm = np.mean(self.data[:, 0])
        ym = np.mean(self.data[:, 1])
        x = self.data[:, 0] - xm
        y = self.data[:, 1] - ym
        X = np.stack([x**2, 2*x*y, -2*x, -2*y, -np.ones((np.size(x), ))], axis=1)
        Y = -y**2
        if self.weighted:
            self.Phi = np.linalg.inv(
                X.T.dot(np.diag(self.W)).dot(X)
                ).dot(
                    X.T.dot(np.diag(self.W)).dot(Y)
                    )
        else:
            try:
                self.Phi = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
            except:
                self.Phi = -1*np.ones(5, )
        try:
            x0=(self.Phi[2]-self.Phi[3]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            y0=(self.Phi[0]*self.Phi[3]-self.Phi[2]*self.Phi[1])/((self.Phi[0])-(self.Phi[1])**2)
            term2=np.sqrt(((1-self.Phi[0])**2+4*(self.Phi[1])**2))
            term3=(self.Phi[4] + (y0)**2 + (x0**2)*self.Phi[0] + 2*self.Phi[1])
            term1=1+self.Phi[0]
            b=(np.sqrt(2*term3/(term1+term2)))
            a=(np.sqrt(2*term3/(term1-term2)))
            alpha=0.5*np.arctan2(2*self.Phi[1],1-self.Phi[0])
            model = [x0+xm, y0+ym, a, b, -alpha]
        except:
            print('Inappropriate model generated')
            model = [np.nan, np.nan, np.nan, np.nan, np.nan]
        if np.all(np.isreal(model)) and np.all(~np.isnan(model)) and np.all(~np.isinf(model)):
            model = model
        else:
            model = [-1, -1, -1, -1, -1]
        return model

    def fit_error(self, data):
        # General purpose function to find the residual
        # model: xc, yc, a, b, theta
        term1 = (data[:, 0] - self.model[0])*np.cos(self.model[-1])
        term2 = (data[:, 1] - self.model[1])*np.sin(self.model[-1])
        term3 = (data[:, 0] - self.model[0])*np.sin(self.model[-1])
        term4 = (data[:, 1] - self.model[1])*np.cos(self.model[-1])
        res = (1/self.model[2]**2)*(term1 - term2)**2 + \
            (1/self.model[3]**2)*(term3 + term4)**2 - 1
        return np.abs(res)

class ransac():
    def __init__(self, data, model, n_min, mxIter, Thres, n_good):
        self.data = data
        self.num_pts = data.shape[0]
        self.model = model
        self.n_min = n_min
        self.D = n_good if n_min < n_good else n_min
        self.K = mxIter
        self.T = Thres
        self.bestModel = self.model(**{'data': data}) #Fit function all data points

    def loop(self):
        i = 0
        if self.num_pts > self.n_min:
            while i <= self.K:
                # Pick n_min points at random from dataset
                inlr = np.random.choice(self.num_pts, self.n_min, replace=False)
                loc_inlr = np.in1d(np.arange(0, self.num_pts), inlr)
                outlr = np.where(~loc_inlr)[0]
                potModel = self.model(**{'data': self.data[loc_inlr, :]})
                listErr = potModel.fit_error(self.data[~loc_inlr, :])
                inlr_num = np.size(inlr) + np.sum(listErr < self.T)
                if inlr_num > self.D:
                    pot_inlr = np.concatenate([inlr, outlr[listErr < self.T]], axis=0)
                    loc_pot_inlr = np.in1d(np.arange(0, self.num_pts), pot_inlr)
                    betterModel = self.model(**{'data': self.data[loc_pot_inlr, :]})
                    if betterModel.error < self.bestModel.error:
                        self.bestModel = betterModel
                i += 1
        else:
            # If the num_pts <= n_min, directly return the model
            self.bestModel = self.model(**{'data': self.data})
        return self.bestModel

# Helper functions
def rotation_2d(theta):
    # Return a 2D rotation matrix in the anticlockwise direction
    c, s = np.cos(theta), np.sin(theta)
    H_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])
    return H_rot

def trans_2d(cx, cy):
    H_trans = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1]])
    return H_trans

def scale_2d(sx, sy):
    H_scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1]])
    return H_scale

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def transformPoints(x, y, H):
    N = np.size(x)
    pts = np.stack([x, y, np.ones(N, )], axis=1) if (N > 1) else np.array([x, y, 1])
    pts = H.dot(pts.T)
    ox = pts[0, :] if N>1 else pts[0]
    oy = pts[1, :] if N>1 else pts[1]
    return (ox, oy)

def fillHoles(I):
    # Fill holes in mask
    x_hole, y_hole = np.where(I == 0)
    for x, y in zip(x_hole, y_hole):
        # Fill hole with the mean value
        opts = I[x-2:x+2, y-2:y+2].reshape(-1)
        if (not isinstance(opts, list)) & (opts.size is not 0) & (sum(opts) != 0):
            I[x, y] = np.round(np.mean(opts[opts!=0]))
    return I

def one_hot2dist(posmask):
    # Input: Mask. Will be converted to Bool.
    h, w = posmask.shape
    mxDist = np.sqrt((h-1)**2 + (w-1)**2)
    if np.any(posmask):
        assert len(posmask.shape) == 2
        res = np.zeros_like(posmask)
        posmask = posmask.astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        res = res/mxDist
    else:
        # No valid element exists for that category
        res = np.zeros_like(posmask)
    return res

def label2onehot(Label):
    Label = (np.arange(4) == Label[..., None]).astype(np.uint8)
    Label = np.rollaxis(Label, 2)
    return Label

def clean_mask(mask):
    '''
    Input: HXWXC mask
    Output: Cleaned mask
    cleans the mask by contraction and dilation of edges maps
    '''
    outmask = np.zeros_like(mask)
    classes_available = np.unique(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for cls_idx in np.nditer(classes_available):
        I = 255*np.uint8(mask == cls_idx)
        I = cv2.erode(I, kernel, iterations=1)
        I = cv2.dilate(I, kernel, iterations=1)
        outmask[I.astype(np.bool)] = cls_idx
    return outmask

def simple_string(ele):
    '''
    ele: String which needs to be stripped of all special characters and moved
    to lower subscript
    Returns a simplified string
    '''
    if type(ele) is list:
        ele = [''.join(e.lower() for e in str(string_ele) if e.isalnum()) for string_ele in ele]
    else:
        ele = ''.join(e.lower() for e in str(ele) if e.isalnum())
    return ele

def pad2Size(img, mask, elParam, pupil_c, size):
    badPup = True if np.all(elParam[0] == -1) else False
    badIri = True if np.all(elParam[1] == -1) else False
    badPup_c = True if np.all(pupil_c == -1) else False

    r_in, c_in = img.shape
    r_out, c_out = size
    up_r = r_out - r_in
    up_c = c_out - c_in

    assert up_r%2 == 0 # The difference needs to be div by 2
    assert up_c%2 == 0

    up_r = int(0.5*up_r)
    up_c = int(0.5*up_c)

    img = np.pad(img, ((up_r, up_r), (up_c, up_c)), mode='constant')
    mask = np.pad(mask, ((up_r, up_r), (up_c, up_c)), mode='constant')

    elParam[0][:2] = elParam[0][:2] + np.array([up_c, up_r]) if not badPup else elParam[0][:2]
    elParam[1][:2] = elParam[1][:2] + np.array([up_c, up_r]) if not badIri else elParam[1][:2]
    pupil_c = pupil_c + np.array([up_c, up_r]) if not badPup_c else pupil_c
    return img, mask, pupil_c, elParam,

def linVal(x, xlims, ylims, offset):
    '''
    Given xlims (x_min, x_max) and ylims (y_min, y_max), i.e, start and end,
    compute the value of y=f(x). Offset contains the x0 such that for all x<x0,
    y is clipped to y_min.
    '''
    if x < offset:
        return ylims[0]
    elif x > xlims[1]:
        return ylims[1]
    else:
        y = (np.diff(ylims)/np.diff(xlims))*(x - offset)
        return y.item()

def getValidPoints(LabelMat):
    '''
    Given labels, identify pupil and iris points.
    pupil: label == 3, iris: label ==2
    '''
    im = np.uint8(255*LabelMat.astype(np.float32)/LabelMat.max())
    edges = cv2.Canny(im, 50, 100) + cv2.Canny(255-im, 50, 100)
    r, c = np.where(edges)
    pupilPts = []
    irisPts = []
    for loc in zip(c, r):
        temp = LabelMat[loc[1]-1:loc[1]+2, loc[0]-1:loc[0]+2]
        condPupil = np.any(temp == 0) or np.any(temp == 1) or temp.size==0 # Not a valid pupil point
        condIris = np.any(temp == 0) or np.any(temp == 3) or temp.size==0
        pupilPts.append(np.array(loc)) if not condPupil else None
        irisPts.append(np.array(loc)) if not condIris else None
    pupilPts = np.stack(pupilPts, axis=0) if len(pupilPts) > 0 else []
    irisPts = np.stack(irisPts, axis=0) if len(irisPts) > 0 else []
    return pupilPts, irisPts

def stackall_Dict(D):
    for key, value in D.items():
        if type(D[key]) is list:
            print('Stacking: {}'.format(key))
            D[key] = np.stack(value, axis=0)
        elif type(D[key]) is dict:
            stackall_Dict(D[key])
    return D

def extract_datasets(subsets):
    '''
    subsets: contains an array of strings
    '''
    ds_idx = [str(ele).split('_')[0] for ele in np.nditer(subsets)]
    ds_present, ds_id = np.unique(ds_idx, return_inverse=True)
    return ds_present, ds_id

def get_ellipse_info(param, H, cond):
    '''
    Parameters
    ----------
    param : np.array
        Given ellipse parameters, return the following:
            a) Normalized Phi values
            b) Points along periphery
            c) Condition to indicate if ellipse exists
    H: np.array 3x3
        Normalizing matrix which converts ellipse to normalized coordinates
    Returns
    -------
    normParam: Normalized Ellipse parameters
    elPts: Points along ellipse periphery
    '''
    if not cond:
        norm_param = my_ellipse(param).transform(H)[0][:-1] # We don't want the area
        elPts = my_ellipse(norm_param).generatePoints(50, 'equiAngle') # Regular points
        elPts = np.stack(elPts, axis=1)
        
        if norm_param[2] > norm_param[3]:
            # This rotates the ellipse by 90 degrees to ensure param 3 is 
            # always greater than 2
            norm_param[[2, 3]] = norm_param[[3, 2]] # Exchange major and minor axis
            norm_param[-1] = np.unwrap(0.5*np.pi + norm_param[-1])
    else:
        # Ellipse does not exist
        norm_param = -np.ones((5, ))
        elPts = -np.ones((8, 2))
    return elPts, norm_param

# Data extraction helpers

def generateEmptyStorage(name, subset):
    '''
    This file generates an empty dictionary with
    all relevant fields. This helps in maintaining
    consistency across all datasets.
    '''
    Data = {k:[] for k in ['Images', # Gray image
                           'dataset', # Dataset
                           'subset', # Subset
                           'resolution', # Image resolution
                           'archive', # H5 file name
                           'Info', # Path to original image
                           'Masks', # Mask
                           'Masks_noSkin', # Mask with only iris and pupil
                           'Fits', # Pupil and Iris fits
                           'pupil_loc']}
    Data['Fits'] = {k:[] for k in ['pupil', 'iris']}

    Key = {k:[] for k in [  'dataset',# Dataset
                            'subset', # Subset
                            'resolution', # Image resolution
                            'archive', # H5 file name
                            'Info', # Path to original image
                            'Fits', # Pupil and Iris fits
                            'pupil_loc']}
    Key['Fits'] = {k:[] for k in ['pupil', 'iris']}
    Data['dataset'] = name
    Data['subset'] = subset
    Key['dataset'] = name
    Key['subset'] = subset
    return Data, Key
