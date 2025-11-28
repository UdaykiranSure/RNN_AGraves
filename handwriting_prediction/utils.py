import torch

def bivariate_gausssain(x,y,mux,muy, varx, vary, corr):
    """
    params: 
        x: (B * L * 1)
        y: (B * L * 1)
        mux: (B * L * M)
        muy: (B * L * M)
        varx:(B * L * M)
        vary:(B * L * M)
        corr:(B * L * M)

    output:
        prob : (B * L * M)
    """

    x_hat = (x - mux) / torch.sqrt(varx)                   # (B * L * M)
    y_hat = (y - muy) / torch.sqrt(vary)                   # (B * L * M)
    z = x_hat**2 + y_hat**2 - (2*corr * x_hat * y_hat)  # (B * L * M)
    t1 = 2.0 * torch.pi * torch.sqrt(varx*vary) * torch.sqrt(1-corr**2)
    t2 = torch.exp(-z / (2.0*(1-corr*2))) 
    N = t2 / t1
    return N                                            #(B * L * M)


# B, L, M = 2, 5, 3
# x = np.random.randn(B, L, 1)
# y = np.random.randn(B, L, 1)
# mux = np.random.randn(B, L, M)
# muy = np.random.randn(B, L, M)
# varx = np.abs(np.random.rand(B, L, M)) + 1e-3
# vary = np.abs(np.random.rand(B, L, M)) + 1e-3
# corr = np.clip(np.random.uniform(-0.9, 0.9, size=(B, L, M)), -0.99, 0.99)

# N = bivariate_gausssain(x, y, mux, muy, varx, vary, corr)
# print("N.shape =", N.shape)  # should be (B, L, M)
# print("sample values:", N.flat[:6])