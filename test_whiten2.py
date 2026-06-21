from core.decomposition.eigenfaces import Eigenfaces
ef = Eigenfaces(n_components=50)
ef.pca.whiten = True
print(f"Whitening: {ef.pca.whiten}")
