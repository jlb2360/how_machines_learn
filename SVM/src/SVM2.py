import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class SVM2():
    def __init__(self):
        self.alphas = None
        self.C = None
        self.K = None
        self.y = None
        self.X = None
        self.errors = None
        self.n_samples = None
        self.n_features = None
        self.b = None
        self.w = None
        self.support_vector_alphas = None
        self.support_vectors_X = None
        self.support_vectors_y = None
        self.tol = None


    def fit(self, X, y, K=linear_kernel, C=1, tol=1E-3, max_pass = 5):
        self.X = X
        self.y = y
        self.C = C
        self.K = K
        self.tol = tol
        self.n_samples, self.n_features = X.shape
        self.alphas = np.zeros(self.n_samples)
        self.errors = np.zeros(self.n_samples)
        self.b = 0
        self.max_pass = max_pass

        self._compute_w()

        # pre-compute errors
        for i in range(self.n_samples):
            self.errors[i] = self._u(i) - self.y[i]

        examine_all = True
        alphas_changed = 0
        passes = 0
        while passes < max_pass and (alphas_changed > 0 or examine_all):
            alphas_changed = 0
            if examine_all:
                for i2 in range(self.n_samples):
                    alphas_changed += self._examine(i2)
            else:
                non_bound_alphas = [i for i, alpha in enumerate(self.alphas) if alpha != 0  or alpha != self.C]
                for i2 in non_bound_alphas:
                    alphas_changed += self._examine(i2)

            if examine_all:
                examine_all = False
            elif alphas_changed == 0:
                examine_all = True

            passes += 1

        self._compute_w()

        sv_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors_X = self.X[sv_indices]
        self.support_vectors_y = self.y[sv_indices]
        self.support_vector_alphas = self.alphas[sv_indices]


    def _examine(self, i2):
        E2 = self.errors[i2]
        alpha2 = self.alphas[i2]
        r2 = E2*self.y[i2]

        non_bound_alphas = [i for i, alpha in enumerate(self.alphas) if alpha != 0 or alpha != self.C]
        
        if (r2 < -self.tol and alpha2 < self.C) or \
            (r2 > self.tol and alpha2 > 0):
            # heuristic 1
            if len(non_bound_alphas) > 1:
                i1 = -1
                max_E = 0
                for i1_hold in non_bound_alphas:
                    if i1_hold == i2: continue

                    if np.abs(self.errors[i1_hold] - E2) > max_E:
                        max_E = np.abs(self.errors[i1_hold] - E2)
                        i1 = i1_hold

                if i1 != -1 and self._min_alpha(i1, i2):
                    return 1



            # heuristic 2
            rand_int = np.random.randint(0, self.n_samples)
            for i in range(self.n_samples):
                i1 = (i + rand_int) % self.n_samples
                if i1 == i2: continue
                if 0 < self.alphas[i1] < self.C:
                    if self._min_alpha(i1, i2):
                        return 1

            # heuristic 3
            rand_int = np.random.randint(0, self.n_samples)
            for i in range(self.n_samples):
                i1 = (i + rand_int) % self.n_samples
                if i1 == i2: continue
                if self._min_alpha(i1, i2):
                    return 1


        return 0


    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            pred=0
            for sv_idx in range(len(self.support_vector_alphas)):
                alpha_sv = self.support_vector_alphas[sv_idx]
                y_sv = self.support_vectors_y[sv_idx]
                x_sv = self.support_vectors_X[sv_idx]
                pred += alpha_sv * y_sv * self.K(x_sv, X_test[i])
            y_pred[i] = np.sign(pred - self.b)
        return y_pred

        
    
    def _min_alpha(self, i1, i2):
        if i1 == i2: return 0
        alpha2_old = self.alphas[i2]
        alpha1_old = self.alphas[i1]
        y1 = self.y[i1]
        y2 = self.y[i2]
        x1 = self.X[i1]
        x2 = self.X[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]

        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old) 
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha2_old + alpha1_old - self.C)
            H = min(self.C, alpha2_old + alpha1_old)

        if L == H:
            return 0

        eta = self.K(x1, x1) + self.K(x2, x2) - 2.0 * self.K(x1, x2)
        if eta == 0.0:
            return 0

        alpha2_unclipped = alpha2_old + (y2 * (E1 - E2))/eta

        if alpha2_unclipped > H:
            alpha2 = H
        elif alpha2_unclipped < L:
            alpha2 = L
        else:
            alpha2 = alpha2_unclipped

        if np.abs(alpha2 - alpha2_old) < self.tol*(alpha2 + alpha2_old + self.tol):
            return 0

        alpha1 = alpha1_old + y1*y2*(alpha2_old - alpha2)

        b1 = E1 + y1*(alpha1 - alpha1_old)*self.K(x1, x1) + y2*(alpha2 - alpha2_old)*self.K(x1, x2) + self.b
        b2 = E2 + y1*(alpha1 - alpha1_old)*self.K(x1, x2) + y2*(alpha2 - alpha2_old)*self.K(x2, x2) + self.b

        if 0 > alpha1  or alpha1 > self.C:
            self.b = b1
        elif 0 > alpha2 or alpha2 > self.C:
            self.b = b2
        else: # Both new alphas are on bounds
            self.b = (b1 + b2) / 2.0


        self.alphas[i1] = alpha1
        self.alphas[i2] = alpha2

        self._compute_w()

        self._cache_error(i1)
        self._cache_error(i2)
        for k in range(self.n_samples):
            if 0 < self.alphas[k] < self.C and k != i1 and k != i2:
                self._cache_error(k)

        return 1


    def _compute_w(self):
        self.w = np.zeros(self.n_features)
        for i in range(self.n_samples):
            if self.alphas[i] > 0.0:
                self.w += self.y[i] * self.alphas[i] * self.X[i]

    def _u(self, i):
        res = 0
        for j in range(self.n_samples):
            if self.alphas[j] > 0:
                res += self.alphas[j] * self.y[j] * self.K(self.X[j], self.X[i])
        
        return res - self.b
    
    def _cache_error(self, i):
        self.errors[i] = self._u(i) - self.y[i]

