#!/usr/bin/env python
# coding: utf-8

# In[45]:


# Initialize Otter
import otter
grader = otter.Notebook("assignment4.ipynb")


# # Final Project <a class='tocSkip'>
#     
# ## PSTAT 134/234 (Winter 2023) <a class='tocSkip'>
#    

# In[46]:


get_ipython().run_line_magic('xmode', 'Verbose')


# <!-- BEGIN QUESTION -->
# 
# ## Question 1: Using Linear Algebra for Optimization
# In recommender system module, low-rank matrix factorization was used to execute latent factor modeling of movie ratings data.
# 
# Specifically, we calculated matrices $U$ and $V$ to solve the following optimization problem (if all ratings were given):
# $$
# \begin{aligned}
# \min_{U,V} f(U,V) &= \min_{U,V} \|R - V U^T\|_F^2
# =\min_{U,V} \left\{ \sum_{m=1}^M\sum_{i=1}^I I_{mi}(r_{mi} - v_m u_i^T)^2 \right\},
# \end{aligned}
# $$
# where
# $$
# \begin{aligned}
# I_{mi} = \begin{cases}
# 1 \text{, if $r_{mi}$ is observed}\\
# 0 \text{, if $r_{mi}$ is missing.}\\
# \end{cases}
# \end{aligned}
# $$
# 
# The best $U$ and $V$ were calculated iteratively by improving on current estimates:
# $$
# \begin{aligned}
# u_i^{\text{new}} &= u_i + 2\alpha(r_{mi} -  v_m u_i^T)\cdot v_m\\
# v_m^{\text{new}} &= v_m + 2\alpha(r_{mi} -  v_m u_i^T)\cdot u_i,
# \end{aligned}
# $$
# where $\alpha$ is the step-size that is to be chosen by the user. (We won't discuss the role in this class, but treat it as an arbitrary, but given, parameter) 
# 
# We can make calculating the updates more efficient by calculating them with matrix operations. For example, instead of calculating each deviation $\gamma_{mi} = r_{mi} - v_m u_i^T$ separately for all $m=1,2,\dots,M$ and $i=1,2,\dots,I$, matrix $\Gamma$ of all deviations can be computed together using matrix operation _(verify for yourself)_:
# $$\Gamma = R - VU^T$$
# 
# Similarly, updating $U$ and $V$ can be combined into matrix calculations which makes the optimization procedure more efficient.
# 
# First, note that updates for $u_i$, $i=1,2,\dots,I$ can be rewritten as
# $$
# \begin{aligned}
# u_1^{\text{new}} &= u_1 + 2\alpha\gamma_{m1}\cdot v_m\\
# u_2^{\text{new}} &= u_2 + 2\alpha\gamma_{m2}\cdot v_m\\
# \vdots\quad &\qquad\qquad\vdots\\
# u_I^{\text{new}} &= u_I + 2\alpha\gamma_{mI}\cdot v_m.
# \end{aligned}
# $$
# Stacking all $I$ equations into a matrix form, 
# $$
# \begin{aligned}
# U^{\text{new}} &= U + 2\alpha\Gamma_{m-}^T v_m,
# \end{aligned}
# $$
# where $\Gamma_{m-}$ is the $m$-th row of $\Gamma$ (use the notation $\Gamma_{-i}$ for the $i$-th column). When evaluating $U^{\text{new}}$, the latest updated values of $U$, $V$, and $\Gamma$ are used.
# 
# Note that there are $M$ such update equations (one for each $m=1,2,\dots,M$) that can also be combined into one matrix update equation involving matrices $U$, $V$, $\Gamma$ and scalars. As stated earlier, since $\alpha$ is assumed to be an arbitrary step-size parameter, we can replace $\alpha/M$ with $\alpha$.
# 
# ### Question 1a: Using Linear Algebra for Optimization
# 
# Complete the following update equations:
# $$
# \begin{aligned}
# U^{\text{new}} &= U + 2\alpha[\text{some function of }\Gamma][\text{some function of }V]\\
# V^{\text{new}} &= V + 2\alpha[\text{some function of }\Gamma][\text{some function of }U]
# \end{aligned}
# $$
# 
# <!--
# BEGIN QUESTION
# name: q1a
# manual: true
# points: 4
# -->

# We can write the update equations as follows:
# 
# $U^{new}=U+2\alpha\Gamma^{T}_{m}-\upsilon_m = U + 2\alpha(R - VU^T)^T_{m-}\upsilon_m$
# $= U + 2\alpha(R_{m-} - V_{m-}{U^{T}_{m-})\upsilon_m}$
# 
# $V^{new}=V + 2\alpha\Gamma_{-i}U_{-i} = V + 2\alpha(R-VU^T)_{-i}U_{-i}$
# $=V + 2\alpha(R_{-i} - V_{-i}U^T_{-i})U_{-i}$
# 

# <!-- END QUESTION -->
# 
# ### Question 1b: Implementing Updates
# 
# In this problem, you will implement the updates calculated in the previous problem. Define the following three functions:
# 
# * `update_G(R, U, V)`: computes deviation $R-VU^T$
# * `update_U(G, U, V, alpha=0.01)`: calculates update $U^{\text{new}}$
# * `update_V(G, U, V, alpha=0.01)`: calculates update $V^{\text{new}}$
# 
# Each function should only be one line of matrix operations. Three functions is to be applied sequentially, using the most up-to-date estimates of $G$, $U$, and $V$.
# 
# Since some elements of `R` are `np.nan` for any missing ratings, `update_U` and `update_V` functions need to be adjusted by using `numpy.nan_to_num` function where appropriate. The function `numpy.nan_to_num` will let you replace `NaN` to some number, so that missing ratings do not interfere with updates.
# 
# <!--
# BEGIN QUESTION
# name: q1b
# manual: false
# points: 4
# -->

# In[47]:


import numpy as np
import pandas as pd

def update_G(R_, U_, V_):
    
    return R_ - np.dot(V_, U_.T)

def update_U(G_, U_, V_, alpha=0.01):
    G_ = np.nan_to_num(G_)
    return U_ + np.dot((2*alpha*G_.transpose()),V_)


def update_V(G_, U_, V_, alpha=0.01):
    G_ = np.nan_to_num(G_)
    return V_ + np.dot((2*alpha*G_),U_)
    
    

# small test to help debug (keep intact)
np.random.seed(1)

M_ = 5
I_ = 3
K_ = 2

R_ = np.random.rand(M_, I_).round(1)
R_[0, 0] = R_[3, 2] = np.nan
U_ = np.random.rand(I_, K_).round(1)
V_ = np.random.rand(M_, K_).round(1)
G_ = update_G(R_, U_, V_)


# In[48]:


grader.check("q1b")


# ### Question 1c: Construct Optimization Algorithm
# 
# Combine the above functions to implement the optimization algorithm to iteratively compute $U$ and $V$.
# 
# But, first, here are functions that will calculate RMSE and quantify the maximum update (in absolute value) made by `update_U` and `update_V` after they are called.

# In[49]:


def rmse(X):
    """
    Computes root-mean-square-error, ignoring nan values
    """
    return np.sqrt(np.nanmean(X**2))

def max_update(X, Y, relative=True):
    """
    Compute elementwise maximum update
    
    parameters:
    - X, Y: numpy arrays or vectors
    - relative: [True] compute relative magnitudes
    
    returns
    - maximum difference between X and Y (relative to Y) 
    
    """
    if relative:
        updates = np.nan_to_num((X - Y)/Y)
    else:
        updates = np.nan_to_num(X - Y)
            
    return np.linalg.norm(updates.ravel(), np.inf)


# A template for the optimization algorithm is given below. Fill-in the missing portions to complete the algorithm.
# 
# <!--
# BEGIN QUESTION
# name: q1c1
# manual: false
# points: 4
# -->

# In[50]:


def compute_UV(Rdf, K=5, alpha=0.01, max_iteration=5000, diff_thr=1e-3):

    R = Rdf.values
    Rone = pd.DataFrame().reindex_like(Rdf).replace(np.nan, 1) # keep data frame metadata

    M, I = R.shape            # number of movies and users
    U = np.random.rand(I, K)  # initialize with random numbers
    V = np.random.rand(M, K)  # initialize with random numbers
    G = update_G(R, U, V)     # calculate residual

    track_rmse = []
    track_update = []
    for i in range(0, max_iteration): 
        
        Unew = update_U(G, U, V, alpha)
        Gnew = update_G(R, Unew, V)

        Vnew = update_V(Gnew, Unew, V, alpha)
        Gnew = update_G(R, Unew, Vnew)

        track_rmse += [{
            'iteration':i, 
            'rmse': rmse(Gnew),
            'max residual change': max_update(Gnew, G, relative=False)
        }]
        track_update += [{
            'iteration':i, 
            'max update':max(max_update(Unew, U), max_update(Vnew, V))
        }]

        U = Unew
        V = Vnew
        G = Gnew
        
        if track_update[-1]['max update'] < diff_thr:
            break
        
    track_rmse = pd.DataFrame(track_rmse)
    track_update = pd.DataFrame(track_update)
    
    kindex = pd.Index(range(0, K), name='k')
    U = pd.DataFrame(U, index= Rdf.columns, columns=kindex)
    V = pd.DataFrame(V, index=Rdf.index, columns=kindex)
    
    return {
        'U':U, 'V':V,
        'rmse': track_rmse,
        'update': track_update
    }
 
Rsmall = pd.read_pickle('data/ratings_stacked_small.pkl').unstack()

np.random.seed(134) # set seed for tests
output1 = compute_UV(Rsmall, K=10, alpha=0.001)


# In[51]:


grader.check("q1c1")


# Running the function on a different sized problem to check if `compute_UV` adapts to changing problem sizes.
# There is nothing new to do here
# 
# <!--
# BEGIN QUESTION
# name: q1c2
# manual: false
# points: 4
# -->

# In[52]:


# These tests should pass if `compute_UV` works properly
np.random.seed(134) # set seed for tests
output2 = compute_UV(Rsmall.iloc[:7, :5], K=8)


# In[53]:


grader.check("q1c2")


# ### Question 1d: Interpret Diagnostic Plots
# 
# Following figures tell us if the optimization algorithm is working properly.

# In[54]:


import altair as alt
logscale = alt.Scale(type='log', base=10)
fig_rmse =     alt.Chart(output1['rmse'])    .mark_line()    .encode(
        x='iteration:Q', 
        y=alt.Y('rmse:Q', scale=logscale)
    )
fig_max_residual_change =     alt.Chart(output1['rmse'])    .mark_line()    .encode(
        x='iteration:Q', 
        y=alt.Y('max residual change:Q', scale=logscale)
    )
fig_updates =     alt.Chart(output1['update'])    .mark_line()    .encode(
        x='iteration:Q', 
        y=alt.Y('max update:Q', scale=logscale)
    )
alt.vconcat(
    fig_rmse | fig_max_residual_change,
    fig_updates 
)


# <!-- BEGIN QUESTION -->
# 
# By referring back to the function used to calculate the quantities in each figure, describe what each figure is showing and interpret the behavior of the optimization algorithm.
# 
# <!--
# BEGIN QUESTION
# name: q1d
# manual: true
# points: 4
# -->

# The first graph displays the root mean square error (rmse) of the model over iterations. From the graph, we can see that rmse decreases as iterations increase, which means our model is improving over time. The second graph shows the maximum change in residual per iteratiion, which is also decreasing over time, meaning our residuals are getting more consistent. The third graph shows how much the output changed, which also has a decreasing trend, however, there are noticable variations within the graph, but that seems to dwindle down as more iterations pass through. 

# <!-- END QUESTION -->
# 
# 
# 
# ### Question 1e: Analyze Large Dataset
# 
# Following code will analyze a larger dataset:

# In[55]:


# run on larger dataset: ratings for 100 movies 
Rbig = pd.read_pickle('data/ratings_stacked.pkl').unstack().iloc[:100]

np.random.seed(14) # set seed for tests
output4 = compute_UV(Rbig, K=5, alpha=0.001, max_iteration=500)

Rhatbig = output4['V']@output4['U'].T


# In[56]:


fit_vs_obs = pd.concat([
    Rhatbig.rename(columns={'rating':'fit'}),
    Rbig.rename(columns={'rating':'observed'}),
], axis=1).stack().dropna().reset_index()[['fit','observed']]

fit_vs_obs = fit_vs_obs.iloc[np.random.choice(len(fit_vs_obs), 5000)]

alt.Chart(fit_vs_obs).transform_density(
    counts = True,
    density='fit',
    bandwidth=0.01,
    groupby=['observed'],
    extent= [0, 6]
).mark_bar().encode(
    alt.X('value:Q'),
    alt.Y('density:Q'),
    alt.Row('observed:N')
).properties(width=800, height=50)


# <!-- BEGIN QUESTION -->
# 
# Consider the above plot. By reading the code, comment on what the plot is illustrating. What happens when you add `counts=True` to `transform_density`? What can you conclude?
# 
# <!--
# BEGIN QUESTION
# name: q1e
# manual: true
# points: 4
# -->

# The plot shows the distributions of the modeled ratings for each 5 ratings. When adding `counts=True`, we get too see the total count instead of the density.

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# ### Question 1f: Make Recommendation
# 
# What movies would you recommend to `user id` 601? Do you see any similarities to movies the user rated high?
# 
# <!--
# BEGIN QUESTION
# name: q1f
# manual: true
# points: 4
# -->

# In[57]:


user = Rhatbig.iloc[:, 600]
user = user.sort_values(ascending=False).head()
print(user)


# The following movies I would recommend user id 601 are shown above.

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# ## Question 2: Improving the Model
# 
# ### Question 2a: Logistic function 
# 
# Note the reconstructed ratings can be smaller than 1 and greater than 5. To confine ratings to between the allowed range, we can use the logistic function. Logistic function is defined as 
# $$ h(x) = \frac{1}{1+e^{-x}}. $$
# It is straightforward to show the derivative is 
# $$ h'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = h(x)(1-h(x)). $$
# Therefore, we can rescale the ratings from $r_{mi}\in [1, 5]$ to $r_{mi}\in [0, 1]$. Then, we can find the best $U$ and $V$ to optimize the following:
# $$ \min_{U,V} \| R - h(VU^T) \|_F^2 = \sum_{m,i} I_{mi}(r_{mi} - h(v_m u_i^T))^2, $$
# where function $h$ is applied elementwise and 
# $$
# \begin{aligned}
# I_{mi} = \begin{cases}
# 1 \text{, if $r_{mi}$ is observed}\\
# 0 \text{, if $r_{mi}$ is missing.}\\
# \end{cases}
# \end{aligned}
# $$
# 
# Derive new update expressions for the new objective function.
# <!--
# BEGIN QUESTION
# name: q2a
# manual: true
# points: 4
# -->

# $$
# \begin{aligned}
# U^{\text{new}} &= U + 2\alpha[R-(h(VU^T)(1−h(VU^T)][V]\\
# V^{\text{new}} &= V + 2\alpha[R-h((VU^T)(1−h(VU^T)][U]
# \end{aligned}
# $$

# <!-- END QUESTION -->
# 
# ### Quesiton 2b: Implementation
# 
# Implement the update functions in functions below.
# 
# <!--
# BEGIN QUESTION
# name: q2b1
# manual: false
# points: 4
# -->

# In[58]:


def logistic(x):
    """
    Evaluates logistic function
    
    """
    return 1/(1+np.exp(-x))

def update_logistic_G(R_, U_, V_):
    
    return R_ - logistic(V_ @ U_.T) 

def update_logistic_U(G_, U_, V_, alpha=0.01):
    
    logisticVUT = logistic(V_ @ U_.T)            # estimated ratings
    grad = -2 * np.nan_to_num(G_ * logistic(V_@ U_.T) * (1 - logisticVUT)) # gradient direction
    return U_ - alpha * grad.T @ V_   # gradient descent update from U_

def update_logistic_V(G_, U_, V_, alpha=0.01):    
    logisticVUT = logistic(V_ @ U_.T)             # estimated ratings
    grad = -2 * np.nan_to_num(G_ * logisticVUT * (1 - logisticVUT)) # gradient direction
    return V_ - alpha * grad @ U_                     # gradient descent update from V_

# small test to help debug (keep intact)
np.random.seed(1)

M_ = 5
I_ = 3
K_ = 2

R_ = np.random.rand(M_, I_).round(1)
R_[0, 0] = R_[3, 2] = np.nan
U_ = np.random.rand(I_, K_).round(1)
V_ = np.random.rand(M_, K_).round(1)
G_ = update_G(R_, U_, V_)


# In[59]:


grader.check("q2b1")


# Now create a function `compute_logistic_UV` below:
# 
# <!--
# BEGIN QUESTION
# name: q2b2
# manual: false
# points: 4
# -->

# In[61]:


def compute_logistic_UV(Rdf, K=5, alpha=0.01, max_iteration=5000, diff_thr=1e-3):

    R = Rdf.values
    R = (R.copy()-1)/4         # map ratings to between 0 and 1
    Rone = pd.DataFrame().reindex_like(Rdf).replace(np.nan, 1) # keep data frame metadata

    M, I = R.shape                 # number of movies and users
    U = np.random.rand(I, K)-0.5   # initialize with random numbers
    V = np.random.rand(M, K)-0.5   # initialize with random numbers
    G = update_G(R, U, V)          # calculate residual

    track_rmse = []
    track_update = []
    for i in range(0, max_iteration): 
        
        Unew = update_logistic_U(G, U, V, alpha)
        Gnew = update_logistic_G(R, Unew, V)

        Vnew = update_logistic_V(Gnew, Unew, V, alpha)
        Gnew = update_logistic_G(R, Unew, Vnew)

        track_rmse += [{
            'iteration':i, 
            'rmse': rmse(Gnew),
            'max residual change': max_update(Gnew, G, relative=False)
        }]
        track_update += [{
            'iteration':i, 
            'max update':max(max_update(Unew, U), max_update(Vnew, V))
        }]

        U = Unew
        V = Vnew
        G = Gnew
        
        if track_update[-1]['max update'] < diff_thr:
            break
        
    track_rmse = pd.DataFrame(track_rmse)
    track_update = pd.DataFrame(track_update)
    
    kindex = pd.Index(range(0, K), name='k')
    U = pd.DataFrame(U, index= Rdf.columns, columns= kindex)
    V = pd.DataFrame(V, index= Rdf.index, columns= kindex)
    
    return {
        'U':U, 'V':V,
        'rmse': track_rmse,
        'update': track_update
    }

def logistic_rating(U_, V_):
    """
    converts the rating back to 1 to 5 rating
    """
    return( 4*logistic(V_@U_.T) + 1 )
    
np.random.seed(134) # set seed for tests
output3 = compute_logistic_UV(Rsmall, K=10, alpha=0.05)


# In[62]:


grader.check("q2b2")


# ### Question 2c: Analyze a Large Dataset
# 
# Following code will analyze a larger dataset:

# In[63]:


# run on larger dataset: ratings for 100 movies 
Rbig = pd.read_pickle('data/ratings_stacked.pkl').unstack().iloc[:100]

np.random.seed(14) # set seed for tests
output4 = compute_logistic_UV(Rbig, K=5, alpha=0.05, max_iteration=500)

Rhatbig = logistic_rating(output4['U'], output4['V'])


# In[64]:


Rhatbig.min()


# In[65]:


fit_vs_obs_2 = pd.concat([
    Rhatbig.rename(columns={'rating':'fit'}),
    Rbig.rename(columns={'rating':'observed'}),
], axis=1).stack().dropna().reset_index()[['fit','observed']]

fit_vs_obs_2 = fit_vs_obs_2.iloc[np.random.choice(len(fit_vs_obs_2), 5000)]

alt.Chart(fit_vs_obs_2).transform_density(
    density='fit',
    bandwidth=0.01,
    groupby=['observed'],
    extent= [0, 6]
).mark_bar().encode(
    alt.X('value:Q'),
    alt.Y('density:Q'),
    alt.Row('observed:N')
).properties(width=800, height=50)


# <!-- BEGIN QUESTION -->
# 
# Consider the above plot. By reading the code, comment on what the plot is illustrating. How does this plot look different than part 1.e?
# 
# <!--
# BEGIN QUESTION
# name: q2c
# manual: true
# points: 4
# -->

# The plot illustrates the density if values for each rating. We can see that the plot looks different than part 1e in respect to the fact that max of 5 is the limit value. Hence, it is more accurate.

# ---
# 
# To double-check your work, the cell below will rerun all of the autograder tests.

# In[ ]:


grader.check_all()


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[ ]:


# Save your notebook first, then run this cell to export your submission.
grader.export()


#  
