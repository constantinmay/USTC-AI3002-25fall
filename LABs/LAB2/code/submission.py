# -*- coding: utf-8 -*-
# submission.py 提交的代码
import os
import numpy as np
from sklearn.cluster import KMeans

# Constants for numerical stability
EPS_LOG = 1e-300  # Small value for log operations to prevent log(0)
EPS_CLIP = 1e-10  # Small value for clipping to avoid division by zero

def data_preprocess(example: np.ndarray) -> np.ndarray:
    """
    完成数据预处理，需要返回处理后的example字典
    
    预处理策略：
    1. 归一化到[0,1]范围
    """
    img_np = np.array(example["image"], dtype=np.float32)

    # 1. 归一化到[0,1]范围
    img_normalized = img_np / 255.0
    
    # 2. 将图像扩展维度 (28,28) -> (1,28,28)
    img_np_with_channel = np.expand_dims(img_normalized, axis=0)
    
    # 3. 将图像展平为一维数组 (28,28) -> (784,)
    img_np_flat = img_normalized.flatten()
    
    # 4. 将处理后的数据添加到example字典中
    example["image2D"] = img_np_with_channel
    example["image1D"] = img_np_flat

    return example

class PCA:
    """
    使用奇异值分解（SVD）实现的简易PCA类

    属性
    ----
    mean_ : np.ndarray
        训练数据每个特征的均值，形状为 (D,)。
    components_ : np.ndarray
        主成分方向，形状为 (n_components, D)。
    explained_variance_ : np.ndarray
        每个主成分的方差解释量，形状为 (n_components,)。
    explained_variance_ratio_ : np.ndarray
        每个主成分的方差贡献比例。
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray = None  # type: ignore
        self.components_: np.ndarray = None  # type: ignore
        self.explained_variance_: np.ndarray = None  # type: ignore
        self.explained_variance_ratio_: np.ndarray = None  # type: ignore

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit the model with X.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Input data.
        """
        N, D = X.shape
        
        # 1. 计算并保存均值
        self.mean_ = np.mean(X, axis=0)
        
        # 2. 中心化数据
        X_centered = X - self.mean_
        
        # 3. 使用SVD进行分解
        # X_centered = U @ S @ Vt
        # 主成分方向是Vt的前n_components行
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 4. 保存主成分（前n_components个）
        self.components_ = Vt[:self.n_components]
        
        # 5. 计算方差解释量
        # 奇异值与方差的关系: variance = S^2 / (N-1)
        explained_variance = (S ** 2) / (N - 1)
        self.explained_variance_ = explained_variance[:self.n_components]
        
        # 6. 计算方差贡献比例
        total_variance = np.sum(explained_variance)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the dimensionality reduction on X."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before calling transform().")
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X."""
        return self.fit(X).transform(X)

    def save_pretrained(self, path: str) -> None:
        """
        保存PCA参数到文件（npz格式）
        """
        np.savez_compressed(
            path,
            n_components=self.n_components,
            mean_=self.mean_,
            components_=self.components_,
            explained_variance_=self.explained_variance_,
            explained_variance_ratio_=self.explained_variance_ratio_
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "PCA":
        """
        从文件加载PCA参数，返回PCA实例
        """
        data = np.load(path)
        n_components = int(data["n_components"]) if "n_components" in data else data["components_"].shape[0]
        obj = cls(n_components)
        obj.mean_ = data["mean_"]
        obj.components_ = data["components_"]
        obj.explained_variance_ = data["explained_variance_"]
        obj.explained_variance_ratio_ = data["explained_variance_ratio_"]
        return obj
    
class GMM:
    """高斯混合模型（EM）。使用全协方差，数值稳定性通过对角线正则项控制。

    参数
    ----
    n_components : int
        混合成分数量。
    max_iter : int
        最大 EM 迭代次数。
    tol : float
        对数似然相对改变量小于该阈值则停止。
    reg_covar : float
        协方差对角线正则，防止矩阵奇异。
    random_state : int
        随机种子。
    init_kmeans : bool
        是否用 KMeans 初始化均值和权重。
    """

    def __init__(self, n_components: int = 10, max_iter: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6, random_state: int = 42, init_kmeans: bool = True) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_kmeans = init_kmeans
        # learned params
        self.weights_: np.ndarray | None = None  # (K,)
        self.means_: np.ndarray | None = None    # (K, D)
        self.covariances_: np.ndarray | None = None  # (K, D, D)
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float | None = None

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def _estep(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """
        E-step: 计算后验概率（responsibilities）
        
        输入:
            X: 数据矩阵 (N, D)
            
        返回:
            resp: 后验概率矩阵 (N, K)，resp[i,k] 表示样本i属于聚类k的概率
            lower_bound: 下界值（用于判断收敛）
        
        实现E-step:
        1. 计算每个样本在每个高斯分布下的对数概率
        2. 使用log-sum-exp技巧计算归一化的后验概率
        3. 返回responsibilities和下界值
        """
        K = self.n_components
        N, D = X.shape
        log_prob = np.empty((N, K), dtype=X.dtype)
        
        # 计算每个样本在每个高斯分布下的加权对数概率
        for k in range(K):
            # 获取当前高斯分布的参数
            mean_k = self.means_[k]
            cov_k = self.covariances_[k]
            weight_k = self.weights_[k]
            
            # 添加正则化项以确保协方差矩阵正定
            cov_k_reg = cov_k + self.reg_covar * np.eye(D)
            
            # 计算多元高斯分布的对数概率密度
            # log N(x|μ,Σ) = -0.5 * [D*log(2π) + log|Σ| + (x-μ)^T Σ^{-1} (x-μ)]
            try:
                # 使用Cholesky分解来稳定计算
                L = np.linalg.cholesky(cov_k_reg)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                
                # 计算 (x-μ)^T Σ^{-1} (x-μ) = ||L^{-1}(x-μ)||^2
                diff = X - mean_k  # (N, D)
                # 解 L @ y = diff.T，得到 y = L^{-1} @ diff.T
                y = np.linalg.solve(L, diff.T)  # (D, N)
                mahalanobis = np.sum(y ** 2, axis=0)  # (N,)
            except np.linalg.LinAlgError:
                # 如果Cholesky分解失败，使用更稳定的方法
                sign, log_det = np.linalg.slogdet(cov_k_reg)
                if sign <= 0:
                    log_det = D * np.log(self.reg_covar)
                diff = X - mean_k
                try:
                    cov_inv = np.linalg.inv(cov_k_reg)
                except np.linalg.LinAlgError:
                    cov_inv = np.eye(D) / self.reg_covar
                mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
            
            # 计算对数概率
            log_prob_k = -0.5 * (D * np.log(2 * np.pi) + log_det + mahalanobis)
            
            # 加上混合权重的对数
            log_prob[:, k] = log_prob_k + np.log(weight_k + EPS_LOG)
        
        # 使用log-sum-exp技巧计算归一化后验概率
        # log(sum(exp(log_prob))) = max + log(sum(exp(log_prob - max)))
        log_prob_max = np.max(log_prob, axis=1, keepdims=True)
        log_prob_norm = log_prob - log_prob_max
        log_sum_exp = log_prob_max.flatten() + np.log(np.sum(np.exp(log_prob_norm), axis=1))
        
        # 计算responsibilities: resp[i,k] = exp(log_prob[i,k] - log_sum_exp[i])
        resp = np.exp(log_prob - log_sum_exp[:, np.newaxis])
        
        # 确保resp是有效的概率分布
        resp = np.clip(resp, EPS_LOG, 1.0)
        resp /= resp.sum(axis=1, keepdims=True)
        
        # 计算下界（对数似然）
        lower_bound = np.sum(log_sum_exp)
        
        return resp, lower_bound

    def _mstep(self, X: np.ndarray, resp: np.ndarray) -> None:
        """
        M-step: 更新模型参数
        
        输入:
            X: 数据矩阵 (N, D)
            resp: 后验概率矩阵 (N, K)
        
        实现M-step:
        更新以下属性：
        1. self.weights_: 每个聚类的权重（先验概率）
        2. self.means_: 每个高斯分布的均值向量
        3. self.covariances_: 每个高斯分布的协方差矩阵
        """
        N, D = X.shape
        K = self.n_components
        
        # 计算每个聚类的有效样本数 N_k = sum_i(resp[i,k])
        N_k = np.sum(resp, axis=0)  # (K,)
        
        # 避免除以零
        N_k = np.clip(N_k, EPS_CLIP, None)
        
        # 1. 更新权重（混合系数）: π_k = N_k / N
        self.weights_ = N_k / N
        
        # 2. 更新均值: μ_k = (1/N_k) * sum_i(resp[i,k] * x_i)
        self.means_ = (resp.T @ X) / N_k[:, np.newaxis]  # (K, D)
        
        # 3. 更新协方差矩阵
        # Σ_k = (1/N_k) * sum_i(resp[i,k] * (x_i - μ_k)(x_i - μ_k)^T)
        self.covariances_ = np.zeros((K, D, D), dtype=X.dtype)
        
        for k in range(K):
            diff = X - self.means_[k]  # (N, D)
            # 加权外积
            weighted_diff = resp[:, k:k+1] * diff  # (N, D)
            cov_k = (weighted_diff.T @ diff) / N_k[k]
            
            # 添加正则化项确保协方差矩阵正定
            cov_k += self.reg_covar * np.eye(D)
            
            self.covariances_[k] = cov_k

    def fit(self, X: np.ndarray) -> "GMM":
        rng = self._rng()
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        # init
        if self.init_kmeans:
            rng_for_kmeans = np.random.RandomState(self.random_state)
            km = KMeans(n_clusters=self.n_components, n_init=5, random_state=rng_for_kmeans)
            labels = km.fit_predict(X)
            self.means_ = km.cluster_centers_.astype(np.float64)
            self.weights_ = np.array([(labels == k).mean() + 1e-12 for k in range(self.n_components)], dtype=np.float64)
            self.weights_ /= self.weights_.sum()
            self.covariances_ = np.stack([np.cov(X[labels == k].T) if np.any(labels == k) else np.eye(D) for k in range(self.n_components)], axis=0).astype(np.float64)
            for k in range(self.n_components):
                if not np.all(np.isfinite(self.covariances_[k])):
                    self.covariances_[k] = np.eye(D)
        else:
            self.means_ = X[rng.choice(N, size=self.n_components, replace=False)]
            self.weights_ = np.ones(self.n_components, dtype=np.float64) / self.n_components
            self.covariances_ = np.stack([np.eye(D) for _ in range(self.n_components)], axis=0)

        prev_lower = None
        for it in range(1, self.max_iter + 1):
            resp, lower = self._estep(X)
            self._mstep(X, resp)
            self.n_iter_ = it
            
            # Skip convergence check on first iteration
            if prev_lower is not None:
                # Compute relative improvement safely
                if np.isinf(prev_lower) or prev_lower == 0:
                    improvement = np.inf
                else:
                    improvement = (lower - prev_lower) / (abs(prev_lower) + EPS_CLIP)
                
                if improvement < self.tol:
                    self.converged_ = True
                    break
            
            prev_lower = lower
        self.lower_bound_ = lower
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("GMM must be fitted before calling predict_proba().")
        X = np.asarray(X, dtype=np.float64)
        resp, _ = self._estep(X)
        return resp

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
    
    def save_pretrained(self, path) -> None:
        """
        保存GMM模型到指定路径（兼容HuggingFace风格）
        """
        import os
        import pickle
        os.makedirs(path, exist_ok=True)
        
        # 保存模型参数
        model_data = {
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'reg_covar': self.reg_covar,
            'random_state': self.random_state,
            'init_kmeans': self.init_kmeans,
            'weights_': self.weights_,
            'means_': self.means_,
            'covariances_': self.covariances_,
            'converged_': self.converged_,
            'n_iter_': self.n_iter_,
            'lower_bound_': self.lower_bound_
        }
        
        with open(os.path.join(path, "gmm_model.pkl"), "wb") as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def from_pretrained(cls, path) -> "GMM":
        """
        从指定路径加载GMM模型
        """
        import pickle
        with open(os.path.join(path, "gmm_model.pkl"), "rb") as f:
            model_data = pickle.load(f)
        
        gmm = cls(
            n_components=model_data['n_components'],
            max_iter=model_data['max_iter'],
            tol=model_data['tol'],
            reg_covar=model_data['reg_covar'],
            random_state=model_data['random_state'],
            init_kmeans=model_data['init_kmeans']
        )
        
        # 恢复训练后的参数
        gmm.weights_ = model_data['weights_']
        gmm.means_ = model_data['means_']
        gmm.covariances_ = model_data['covariances_']
        gmm.converged_ = model_data['converged_']
        gmm.n_iter_ = model_data['n_iter_']
        gmm.lower_bound_ = model_data['lower_bound_']
        
        return gmm