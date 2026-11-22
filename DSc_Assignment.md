# Data Science — CSE558 Assignment

**Submit by 2359 of 22/11/26 · Maximum marks: 60**

- Form a group of two people and submit your assignment. Only one of the group members should submit the assignment before the deadline.
- You are encouraged to discuss with other groups. Mention their names in your submission. If you have used the internet to understand the solution you are writing, then mention the URL.
- Write answers and code of your own; do not copy from others.
- Prepare one zip file with a PDF of your answers (with all tables and plots) and all your code/python notebooks. Name it "rollno1-rollno2-DSA1", e.g., "24001-24002-DSA1". Submit the zip file through Google Classroom. A late submission loses 50% of the marks.
- Fix your random seeds, so that every number you report can be reproduced.

Best wishes!

---

## Data

Both questions use random projection on the KDD Cup 1999 dataset, from a contest on network intrusion detection. Each row is one network connection, described by 41 features and labelled normal or as one of 22 types of attack. Fetch the default 10% version in your python code with

```python
from sklearn.datasets import fetch_kddcup99
kdd = fetch_kddcup99(subset=None, as_frame=True)
```

`kdd.data` holds the 41 features and `kdd.target` the label of each connection. Build the data matrix $D$ from `kdd.data` as follows.

1. One-hot encode the three text features `protocol_type`, `service` and `flag`, and convert the other 38 features to float.
2. Drop every column that takes the same value on all rows.
3. Rescale each column to $[0,1]$ by $x \mapsto (x - \min)/(\max - \min)$.

Report the number of rows $n$ and of columns $d$ of the resulting $D \in \mathbb{R}^{n \times d}$; row $D_i \in \mathbb{R}^d$ is connection $i$. If memory is tight, keep $D$ in float32. Generate every random matrix yourself with numpy; do not use `sklearn.random_projection`.

---

## 1. JL for k-means clustering

Let $k = 15$. For a set $B$ of $k$ centres in $\mathbb{R}^d$ let

$$\mathrm{cost}(D, B) = \sum_{i=1}^{n} \min_{b \in B} \lVert D_i - b \rVert_2^2,$$

and for a partition $\mathcal{C} = (C_1, \dots, C_k)$ of the rows of a matrix $X$ let

$$\mathrm{cost}_X(\mathcal{C}) = \sum_{j=1}^{k} \sum_{i \in C_j} \lVert X_i - \mu_j(X) \rVert_2^2, \qquad \mu_j(X) = \frac{1}{|C_j|} \sum_{i \in C_j} X_i.$$

Let $A \in \mathbb{R}^{m \times d}$ have i.i.d. $N(0, 1/m)$ entries (mean 0 and variance $1/m$, so standard deviation $1/\sqrt{m}$), and let $E = DA^\top \in \mathbb{R}^{n \times m}$, so row $i$ of $E$ is $AD_i$. Use `sklearn.cluster.KMeans` with `n_clusters=15` and `n_init=5` for every clustering in this question.

$k$-means on $E$ returns clusters $\mathcal{C}_E = (C_1, \dots, C_k)$ and $k$ centres in $\mathbb{R}^m$, which cannot be compared with the rows of $D$. To get centres in $\mathbb{R}^d$, keep the clusters and average the original rows of $D$ in each:

$$\hat{b}_j = \frac{1}{|C_j|} \sum_{i \in C_j} D_i \quad (j = 1, \dots, k), \qquad \hat{B} = {\hat{b}_1, \dots, \hat{b}_k}.$$

Then $\mathrm{cost}(D, \hat{B})$ follows from the definition above: assign each row of $D$ to its nearest centre in $\hat{B}$ and add the squared distances.

**(a)** Run $k$-means on $D$; call its centres $B$ and its clusters $\mathcal{C}_D$. Report $\mathrm{cost}(D, B)$ and the wall-clock time of the run. **(3)**

**(b)** For each $m \in {5, 10, 15, 20, 25}$, draw $A$, compute $E$, run $k$-means on $E$ and form $\hat{B}$. Report $\mathrm{cost}(D, \hat{B})$ and the ratio $\mathrm{cost}(D, \hat{B}) / \mathrm{cost}(D, B)$. Repeat this 5 times for each $m$, with a fresh $A$ each time, and give the mean, smallest and largest ratio for each $m$ in a table or bar graph. Comment on how the ratio changes with $m$, and on any ratio below 1. **(10)**

**(c)** In the same runs, also map each centre $a \in \mathbb{R}^m$ found on $E$ back to $\mathbb{R}^d$ as $A^{+}a$, where $A^{+}$ is the pseudo-inverse of $A$ (`numpy.linalg.pinv`); call these $k$ centres $B^{+}$. Report $\mathrm{cost}(D, B^{+}) / \mathrm{cost}(D, B)$ next to the ratio in (b). Which of $\hat{B}$ and $B^{+}$ would you use? **(4)**

**(d)** For each $m$, report the wall-clock time of computing $E$ plus running $k$-means on $E$, averaged over the 5 runs in (b). Plot these times against $m$, with the time from (a) as a horizontal line. **(4)**

**(e)** Pick $10^4$ random pairs $(i, l)$ with $D_i \neq D_l$, and use the same pairs for every $m$. For each $m$:

- (i) for one draw of $A$, compute $\lVert E_i - E_l \rVert_2^2 / \lVert D_i - D_l \rVert_2^2$ for these pairs;
- (ii) for 20 independent draws of $A$, compute $\mathrm{cost}_E(\mathcal{C}_D) / \mathrm{cost}_D(\mathcal{C}_D)$ for the clusters $\mathcal{C}_D$ from (a).

Show the ratios from (i) and from (ii) as box plots against $m$, in one figure. Which of the two has the smaller spread, and how does each change with $m$? **(9)**

---

## 2. JL for linear regression

The label of connection $i$ is `kdd.target[i]`, a bytes string such as `b'normal.'` or `b'smurf.'`. Let $y \in {0,1}^n$ with $y_i = 1$ if connection $i$ is an attack (its label is not `b'normal.'`) and $y_i = 0$ otherwise.[^1] In python,

```python
y = (kdd.target != b'normal.').to_numpy(dtype=float)
```

We fit $y$ by least squares: $b$ minimises

$$\mathrm{Loss}(D, y, a) = \lVert Da - y \rVert_2^2.$$

No intercept column is needed. Solving for $b$ reads all $n$ rows. Instead, we sketch the rows: for a random $S \in \mathbb{R}^{m \times n}$ with $m \ll n$, compute $SD$ and $Sy$, and let $\tilde{a}$ minimise $\lVert SDa - Sy \rVert_2^2$. Use `numpy.linalg.lstsq` for every least-squares problem in this question.

Multiplying $D$ by a dense random $S$ is expensive, so we use a sparse JL matrix (CountSketch). For each $i \in {1, \dots, n}$, draw a bucket $h(i)$ uniformly from ${1, \dots, m}$ and a sign $\sigma(i)$ uniformly from ${-1, +1}$, all independently, and set $S_{h(i), i} = \sigma(i)$, with every other entry 0. Thus $SD$ is formed in one pass over $D$: row $i$ of $D$ is added, with sign $\sigma(i)$, to row $h(i)$ of $SD$. This is the hashing of Count-Min, with random signs. Never store $S$ as a dense matrix: add the rows directly (e.g., with `numpy.add.at`).

**(a)** Report $\mathrm{Loss}(D, y, b)$ and the time to compute $b$. **(3)**

**(b)** For each of $m = 50, 100, 200, 500, 1000, 2000$ and $5000$, draw $S$, compute $\tilde{a}$, and report $\mathrm{Loss}(D, y, \tilde{a})$, its ratio to $\mathrm{Loss}(D, y, b)$, and the sketched loss $\lVert SD\tilde{a} - Sy \rVert_2^2$. Repeat this 5 times for each $m$, with a fresh $S$ each time. Plot the mean ratio against $m$ on log–log axes, with the smallest and largest of the 5 ratios as error bars. Describe what happens to the ratio and to the sketched loss below and above $m = \mathrm{rank}(D)$. **(12)**

**(c)** For each $m$, report the time to draw $S$, form $SD$ and $Sy$ and compute $\tilde{a}$, averaged over the 5 runs in (b), and compare it with the time to compute $b$ in (a). **(4)**

**(d)** Let $v = Db - y$ be the residual of the full solution. For $m \in {50, 500, 5000}$, draw 100 independent CountSketches $S$ and compute $\lVert Sv \rVert_2^2 / \lVert v \rVert_2^2$. For each $m$, plot a histogram and report the mean and standard deviation. How does the spread change with $m$? **(5)**

**(e)** Select 50,000 rows uniformly at random without replacement, once, using a fixed random seed. Use this same subset for both sketching methods and all five repetitions. Draw a fresh sketch in each repetition. Let $D'$ hold these rows and $y'$ the matching entries of $y$, and let $b'$ minimise $\mathrm{Loss}(D', y', a)$. For $m = 500$, sketch $(D', y')$ in two ways:

- (i) with a CountSketch, and
- (ii) with a dense $S \in \mathbb{R}^{m \times 50000}$ with i.i.d. $N(0, 1/m)$ entries.

For each, solve the sketched problem for $\tilde{a}$, and report the time to draw $S$ and form $SD'$ and $Sy'$, and the ratio $\mathrm{Loss}(D', y', \tilde{a}) / \mathrm{Loss}(D', y', b')$, each averaged over 5 repetitions. Which sketch would you use on the full $D$? **(6)**

[^1]: The label itself takes 23 values and is nominal, so its codes cannot serve as a numeric response.