from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    q1 = np.random.normal(10,1,1000)
    x1 = UnivariateGaussian()
    x1.fit(q1)
    print("("+str(x1.mu_)+","+str(x1.var_)+")")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10,1000,100).astype(int)
    estimate = np.zeros(100)
    for i in range(1,estimate.size):
        estimate[i] = np.abs(x1.fit(q1[:i*10]).mu_-10)

    go.Figure([go.Scatter(x=ms, y=estimate, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{absolute distance between the estimate and "
                        r"the true value of the expectation}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$\mu$",height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    x1.fit(q1)
    go.Figure([go.Scatter(x=q1, y=x1.pdf(q1), mode='markers',
                          name=r'$\widehat\PDF$')],
              layout=go.Layout(
                  title=r"$\text{sample value and their PDFs}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$PDF$")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    sigma = np.array([[1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]])
    q2 = np.random.multivariate_normal(mu, sigma, 1000)
    x2 = MultivariateGaussian()
    x2.fit(q2)
    print("estimated expectation: ")
    print(x2.mu_)
    print("covariance: ")
    print(x2.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.array(np.linspace(-10,10,200))
    f3 = np.array(np.linspace(-10,10,200))
    log_likely = np.array([[MultivariateGaussian.log_likelihood(
        np.array([i1,0,i3,0]), sigma, q2) for i3 in f3] for i1 in f1])
    np.reshape(log_likely,(200,200))
    fig = go.Figure(data=go.Heatmap(x=f1, y=f3, z=np.array(log_likely)),
                   layout=go.Layout(title="log-likelihood heatmap of models"
                                          " with different expectation"))
    fig.show()

    # Question 6 - Maximum likelihood
    max1,max3 = np.unravel_index(log_likely.argmax(),log_likely.shape)
    print(f1[max1], f3[max3])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


