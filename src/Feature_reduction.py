from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def feture_reduction_pca (X):
    pca = PCA(n_components=0.9, copy=True)
    pca.fit(X)
    X_pca = pca.transform(X)
    """print(f"Components:\n{pca.components_}")
    print(f"Explained Variance:\n{pca.explained_variance_}")
    print(f"Explained Variance Ratio:\n{pca.explained_variance_ratio_}")
    print(f"Sum of Exmplained Variance Ratio:\n{sum(pca.explained_variance_ratio_)}")"""
    return X_pca

def feture_reduction_tsne(X):
    tsne = TSNE(n_components=0.9, n_iter=2000)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

    """x_train, x_test, y_train, y_test = train_test_split(transformed_x, y, random_state=42, test_size=0.30)
"""

