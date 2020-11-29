
# Computes the Gram matrix given phi_j(x)
#  "The Gram matrix can be computed efficiently by reshaping phi_j(x) into a matrix psi of
# shape C_j × H_jW_j ; then G^phi_j(x) = psi psi^T / C_jH_jW_j"
# https://arxiv.org/pdf/1603.08155.pdf
def gram_matrix(x):
    (j, c, h, w) = x.size()
    psi = x.view(j, c, h*w)
    G = psi.bmm(psi.tranpose(1 ,2)) / (c * h * w)
    return G