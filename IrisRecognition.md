# IrisRecognition
### Result
#### CRR
The CRR table is as follows:
|Similarity measure|Reduced feature set Correct recognition rate(%)|
|-----|-----------------|
|$L_1$ distance measure|87.037|
|$L_2$ distance measure|87.5|
|Cosine similarity measure|90.509|

The CRR plot is as follws:
![](/ResultImages/CRR_plot.png)
#### ROC
The ROC table is as follows:
|Threshold|False match rate(%)|False non-match rate(%)|
|---|----|--|
|0.1|0|90.49|
|0.2|0|89.72|
|0.3|0|85.25|
|0.4|7.14|74.34|
|0.5|2.44|49.21|
|0.6|6.73|18.75|
|0.7|9.49|0|
|0.8|9.49|0|
|0.9|9.49|0|

Now we could plot the ROC curve:
![](/ResultImages/ROC_plot.png)

### Peer Evaluation Form
Yi Yang(yy3105): IrisLocalization, IrisNormalization, ImageEnhancement

Haosheng Ai(ha2583): IrisLocalization, FeatureExtraction, IrisRecognition

Mengjun Zhu(mz2842): IrisMatching, PerformanceEvaluation, README
## IrisLocalization
The function ``IrisLocalization(images, c1, c2, h1, h2)`` does the following:

It takes a positional argument `images` which is our iris dataset.
``c1, c2, h1, h2`` are parameters used in function ``cv2.Canny`` and ``cv2.HoughCircles``.

We first convert the images into gray images and then use a Bilateral filter to remove the noise.

```python
img = cv2.bilateralFilter(img, 9,75,75)
```
Next we convert the noise-removed images into binary images and then project those images into horizontal and vertical directions and follow the instructions in the paper that use the coordinates corresponding to the minima of the two projection profiles to estimate the center
coordinates of the pupil.
```python
center_x = np.mean(img_binary,0).argmin()
center_y = np.mean(img_binary,1).argmin()
```
Next we then follow the instructions in the paper that binarize a $120\times 120$ region centered at the point ($X_p, Y_p$) and use the centroid of the resulting binary region as a more accurate estimate of the center. We use a for-loop to implement this process twice as discussed in the paper.
```python
center_x_120 = np.mean(img120_binary, 0).argmin()
center_y_120 = np.mean(img120_binary, 1).argmin()
center_x = center_x - 60 + center_x_120
center_y = center_y - 60 + center_y_120
```
Then we use the Canny edge detection in the $120\times 120$ region and apply Hough transformation on the edged image to detect every possible circles.

```python
edged = cv2.Canny(img120, c1, c2)
circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, h1, h2)
```

In the following for-loop, in order to find the pupil circle, we design an algorithm that measure the distance between our estimated center $(X_p,Y_p)$ and the center of every circle detected by Hough transformation. By recording the minimum distance we then find the pupil circle.

```python
k = []
error = float("inf")
for j in circles[0]:
    distance = math.sqrt((j[1] - center_x_120) ** 2 + (j[0] - center_y_120) ** 2)
    if distance <= error:
        error = distance
        k = j
```
Now since there might be bias lying on the circles detected by Hough Transformation, we then combine the circles estimated in the $120\times 120$ region and the Hough circles.
$$Center=\omega Center_{Hough}+(1-\omega)Center_{120}$$
Here $\omega=0.6$,
```python
weight = 0.6
center_x_mix = round(center_x_hough * weight + center_x * (1 - weight))
center_y_mix = round(center_y_hough * weight + center_y * (1 - weight))
```
Now we return the list ``centers`` which contains the location and the radius of the pupil center.

## IrisNormalization
First we apply the algorithm mentioned in the paper:
$$I_n(X,Y)=I_o(x,y)$$
$$x=x_p(\theta)+(x_i(\theta)-x_p(\theta))\frac{Y}{M}$$
$$y=y_p(\theta)+(y_i(\theta)-y_p(\theta))\frac{Y}{M}$$
$$\theta=2\pi X/N$$
```python
def polar(ang,r,cx,cy):
    x = r*np.cos(ang)+cx
    y = r*np.sin(ang)+cy
    return x,y

def IrisNormalization(images,center,dist = 55):
    M = 64
    N = 512
    result= []
    for i in range(len(images)):
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        cx = center[i][0]
        cy = center[i][1]
        r = center[i][2]
        temp = np.zeros((M, N), np.uint8)
        for m in range(M):
            for n in range(N):
                theta = 2 * np.pi * n/N
                xp,yp = polar(theta, r,cx,cy)
                xi,yi = polar(theta, (r+dist), cx, cy)
                x,y = polar(theta, (r+m*dist/M), cx, cy)
                X = n
                if(n == 128*3 or n == 128):
                    Y = M * (y - yp) / (yi - yp)
                else:
                    Y = M * (x - xp) / (xi - xp)      
                if(round(x)>=img.shape[1] or round(y) >= img.shape[0]):
                    continue
                else:
                    temp[round(Y),round(X)] = img[round(y),round(x)]
        result.append(temp)
    return(result)
```
However the results seem not very good. So we improve this algorithm by detect the ring region between the pupil and iris. We transform this region into polar system and record the polar location of the region. Finally, resize this list into a $64\times 512$ image.

The first script is used to divide the region into 200 parts:
```python
n = 200
theta_range = np.arange(n) * 2 * np.pi / n

# divide the radius into 55 parts. 
# 55 is the distance between the pupil radius and the iris radius.
# The distance is observed more often close to 55.
iris_radius = 55
radial_range = range(iris_radius)
```
Now we do the transformation algorithm:
```python
res = np.zeros((iris_radius, n))
for r in radial_range:
    for theta in theta_range:
        x = int((r + radius_pupil) * np.cos(theta) + center_x)
        y = int((r + radius_pupil) * np.sin(theta) + center_y)
        if y >= 280 or x >= 320:
            continue
        else:
        theta_index = int((theta * n) / (2 * np.pi))
        res[r][theta_index] = img[y][x]
# resize the new image into 64*512 size
normalized_images.append(cv2.resize(res, (512, 64)))
```
Finally we return the normalized images ``normalized_images``.
## ImageEnhancement
The function ``ImageEnhancement(images)`` takes a positional argument ``images`` which is the normalized image processed by ``IrisNormalization.``

We use the function ``cv2.equalizeHist`` to implement the enhancement preprocess of the normalized images.
## FeatureExtraction

First we obtain the defined filter lies in the modulating sinusoidal function:
$$M(x, y, f)=\cos\left[2\pi f(\sqrt{x^2+y^2})\right]$$
$$G(x, y, f)=\frac{1}{2\pi\delta_x\delta_y}\exp\left[-\frac{1}{2}(\frac{\delta_x^2}{x^2}+\frac{\delta_y^2}{y^2})\right]M(x, y, f)$$
```python
def M(x, y, f):
    """
    define the modulating sinusoidal function mentioned in the paper.
    """
    m = np.cos(2 * np.pi * f * math.sqrt(x ** 2 + y ** 2))
    return m


def G(x, y, dx, dy, f):
    """
    define the desired filter mentioned in the paper
    """
    g = 1 / (2 * np.pi * dx * dy) * np.exp(-0.5 * (x ** 2/dx ** 2+y ** 2/dy ** 2)) * M(x, y, f)
    return g
```
Next we obtain the filter kernal function defined in the paper in a $9\times 9$ region:
$$F(x,y)=\int\int I(x_1,y_1)G(x-x_1,y-y_1)dx_1dy_1$$
```python
def conv(dx, dy, f):
    """
    define the convolution mask to realize function F defined in the paper
    """
    conv = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            conv[i, j] = G((-4+j), (-4+i), dx, dy, f)
    return conv
```

The function ``feature(conv_res)`` takes a positional argument which is the result of the convolution and then return the feature vector V:
$$V=[m_1,\sigma_1,...]^T$$
where 
$$m=\frac{1}{n}\sum_{w}|F(x,y)|\qquad\sigma=\frac{1}{n}\sum_{w}||F(x,y)|-m|$$

Now in the main function ``FeatureExtraction(enhanced)``, we take the input which is the enhanced images, and then obtain the ROI region by ``roi = img[:48, :]``.

Now for the two channels: $\delta_x=3$, $\delta_y=1.5$ and $\delta_x=4.5$, $\delta_y=1.5$, here we take the frequency $f=1/\delta_y=1/1.5$, we use the function ``scipy.signal.convolve2d`` to compute the convolution between the filter kernal and our ROI region respectively. Then we use our defined function ``feature(conv_res)`` to compute the feature vector and store it in the list ``features`` and return it.

## IrisMatching
Frist we use the LinearDiscriminantAnalysis analysis to reduce dimension to n by defining a ``LDA`` function.

Next obtain the $L_1$, $L_2$ and cosine similarity measures respectively.
```python
def get_l1(feature1, feature2):
    """
    calculate the L1 distance between two input features
    """
    sum_l1 = 0
    for i in range(len(feature1)):
        sum_l1 += abs(feature1[i] - feature2[i])
    return sum_l1


def get_l2(feature1, feature2):
    """
    calculate the L2 distance between two input features
    """
    sum_l2 = 0
    for i in range(len(feature1)):
        sum_l2 += math.pow((feature1[i] - feature2[i]), 2)
    return sum_l2


def get_cosine(feature1, feature2):
    """
    calculate the cosine similarity between two input features
    """
    temp1 = np.linalg.norm(feature1)
    temp2 = np.linalg.norm(feature2)
    cosine = 1 - ((np.matmul(np.transpose(feature1), feature2)) / (temp1*temp2))
    return cosine
```
Now the main function ``IrisMatching(features_train, features_test, n)`` takes three parameters: ``features_train``: the feature of the training images processed by FeatureExtraction; 
    ``features_test``: the feature of the test images processed by FeatureExtraction;
    ``n``: the n_components parameter in the function LinearDiscriminantAnalysis.
We then calculate the $L_1$, $L_2$ and cosine similarity measures and record the minimizers for the three measures respectively. Also we record the minimum cosine similarity measures.
We return four values ``res_L1, res_L2, res_cosine, min_cosine``:
``res_L1, res_L2, res_cosine`` are the indices where the values
    minimize the $L_1$, $L_2$, cosine similarity distances respectively. ``min_cosine`` records the min cosine similarity distance.
## PerformanceEvaluation
First we define a function ``get_test_model_label(res)`` which takes one positional argument ``res`` which is the training result generated by ``IrisMatching``. This is the function that is used to classsify the training result for the three measures respectively. We devide the result values by 3 and get an integer result. This integer is in the range of 0 to 107 and by classifying this value we could know which image belongs to the orginal person.
```python
def get_test_model_label(res):
    """
    Compute the label of test images for the input match result (L1, L2 and cosine similarity)
    """
    labels = []
    for i in range(len(res)):
        temp = res[i]
        labels.append(temp // 3)
    return np.array(labels)
```
Now we want to calculate the ROC, we first to set a threshold value. If the min cosine similarity is less than this threshold, we record this location by 1. Otherwise we then record 0.
```python
def get_cosine_label(threshold, res):
    """
    Compute the label of test images for the input cosine similarity
    """
    labels = []
    for i in range(len(res)):
        if res[i] < threshold:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)
```
The function ``get_test_real_label()`` is simply used to record the real labels for the testing dataset.

Now we define our main function ``PerformanceEvaluation(res_L1, res_L2, res_cosine, min_cosine)``:

``res_L1``:  the minimizer for the L1 distance generated by IrisMatching

``res_L2``: the minimizer for the L2 distance generated by IrisMatching

``res_cosine``: the minimizer for the cosine similarity distance generated by IrisMatching

``min_cosine``: the minimum cosine similarity distance generated by IrisMatching

First we calculate the CRR. We use the logic here that if we have successfully matched the results, then the ``test_labels`` generated by ``get_test_model_label(res)`` should have the same labels with the ``test_real_labels``,
hence we could generate a Boolean list using ``test_labels==test_real_labels``. We calculate the summation of those `True`'s and then divide it by the length of the ``test_labels``, we then obtain the CRR.
```python
crr_L1 = sum((test_labels_L1-test_real_labels) == 0) / len(res_L1)
crr_L2 = sum((test_labels_L2-test_real_labels) == 0) / len(res_L2)
crr_cosine = sum((test_labels_cosine-test_real_labels) == 0) / len(res_cosine)
```
Now regarding the FMR and FNMR. Here the FMR means we should not match the results but mistakenly match the results; And the FNMR means we should match the results but mistakenly do not match the results.

We first record the cosine label vector:
```python
test_labels_cosine_ROC = get_cosine_label(threshold, min_cosine)
```
For the FMR, if it is indeed an FMR error, we then should not match the results but do match. Hence the number that we should match the result is:
```python
sum(test_labels_cosine_ROC == 1)
```
Now we first construct a Boolean list that ``test_labels_cosine`` does not equal ``test_real_labels``.
This means when ``test_labels_cosine != test_real_labels`` .
Here we should match means ``test_labels_cosine_ROC == 1``, so we multiply the Boolean vector by ``test_labels_cosine_ROC == 1`` and hence get the FMR rate:

```python
if sum(test_labels_cosine_ROC == 1) != 0:
    fmr = sum((test_labels_cosine !=
                test_real_labels) * (test_labels_cosine_ROC == 1)) / sum(test_labels_cosine_ROC == 1)
else:
    fmr = 0
```

For the FNMR, if it is indeed an FNMR error, we then should match the results but fail to match. Hence the number that we should not match the result is:
```python
sum(test_labels_cosine_ROC == 0)
```
Now we first construct a Boolean list that ``test_labels_cosine`` does equal ``test_real_labels``.
This means when ``test_labels_cosine == test_real_labels`` .
Here we should not match means ``test_labels_cosine_ROC == 0``, so we multiply the Boolean vector by ``test_labels_cosine_ROC == 0`` and hence get the FMR rate:
```python
 if sum(test_labels_cosine_ROC == 0) != 0:
    fnmr = sum((test_labels_cosine ==
                test_real_labels) * (test_labels_cosine_ROC == 0)) / sum(test_labels_cosine_ROC == 0)
else: 
    fnmr = 0
```

Now return the CRR values and FMR, FNMR values in ``crr_L1, crr_L2, crr_cosine, FMR, FNMR``.