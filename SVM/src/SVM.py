import numpy as np


class SVM():
    def __init__(self, C=1.0, tol=1E-3, max_passes=5, kernel_type='linear'):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

        if kernel_type == 'linear':
            self.kernel = self._linear_kernel
        else:
            raise ValueError("Kernel not supported.")
        

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n_samples, self.n_features = X.shape
        self.alphas = np.zeros(self.n_samples)
        self.b = 0.0

        if hasattr(self, 'kernel') and self.kernel == self._linear_kernel:
            self._compute_w()

        self.errors = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            self.errors[i] = self._decision_function_i(i) - self.y_train[i]

        passes = 0
        num_changed_alphas = 0
        examine_all = True

        while passes < self.max_passes and (num_changed_alphas > 0 or examine_all):
            num_changed_alphas_in_pass = 0
            if examine_all:
                for i in range(self.n_samples):
                    num_changed_alphas_in_pass += self._examine_alpha(i)
            else:
                non_bound_indices = [i for i, alpha in enumerate(self.alphas) if 0 < alpha < self.C]
                for i in non_bound_indices:
                    num_changed_alphas_in_pass += self._examine_alpha(i)
            
            num_changed_alphas = num_changed_alphas_in_pass

            if examine_all:
                examine_all = False
            elif num_changed_alphas_in_pass == 0:
                examine_all = True
            
            passes += 1

        
        self._compute_w()

        sv_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors_X = self.X_train[sv_indices]
        self.support_vectors_y = self.y_train[sv_indices]
        self.support_vector_alphas = self.alphas[sv_indices]


    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            pred=0
            for sv_idx in range(len(self.support_vector_alphas)):
                alpha_sv = self.support_vector_alphas[sv_idx]
                y_sv = self.support_vectors_y[sv_idx]
                x_sv = self.support_vectors_X[sv_idx]
                pred += alpha_sv * y_sv * self.kernel(x_sv, X_test[i])
            y_pred[i] = np.sign(pred + self.b)
        return y_pred

    def _compute_w(self):
        self.w = np.zeros(self.n_features)
        for i in range(self.n_samples):
            if self.alphas[i] > 0:
                self.w += self.alphas[i] * self.y_train[i] * self.X_train[i]

    def _decision_function_i(self, i):
        res = 0
        for j in range(self.n_samples):
            if self.alphas[j] > 0:
                res += self.alphas[j] * self.y_train[j] * self.kernel(self.X_train[j], self.X_train[i])

        return res + self.b
    
    def _examine_alpha(self, i2):
        """
        This function uses three heuristics to pick an alpha1
        """

        y2 = self.y_train[i2]
        alpha2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        if (r2 < -self.tol and alpha2 < self.C) or \
            (r2 > self.tol and alpha2 > 0):

            # heuristic 1
            non_bound_indices = [i for i, alpha in enumerate(self.alphas) if 0 < alpha < self.C]
            if len(non_bound_indices) > 1:
                best_i1 = -1
                max_delta_E = 0
                for i1_hold in non_bound_indices:
                    if i1_hold == i2: continue
                    E1_hold = self.errors[i1_hold]
                    delta_E = np.abs(E1_hold - E2)
                    if delta_E > max_delta_E:
                        max_delta_E = delta_E
                        i1 = i1_hold
                if i1 != -1 and self._take_step(i1, i2):
                    return 1 # alpha changed
                
            # heuristic 2
            rand_start = np.random.randint(0, self.n_samples)
            for i1_offset in range(self.n_samples):
                i1 = (rand_start + i1_offset) % self.n_samples
                if 0 < self.alphas[i1] < self.C:
                    if i1 == i2: continue
                    if self._take_step(i1, i2):
                        return 1 # alpha changed
                    
            # heuristic 3
            rand_start = np.random.randint(0, self.n_samples)
            for i1_offset in range(self.n_samples):
                i1 = (rand_start + i1_offset) % self.n_samples
                if i1 == i2: continue
                if self._take_step(i1, i2):
                    return 1 # alpha changed
                
        return 0
        
    def _take_step(self, i1, i2):
        
        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        y1 = self.y_train[i1]
        y2 = self.y_train[i2]

        E1 = self.errors[i1]
        E2 = self.errors[i2]

        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L >= H:
            return 0
        
        k11 = self.kernel(self.X_train[i1], self.X_train[i1])
        k22 = self.kernel(self.X_train[i2], self.X_train[i2])
        k12 = self.kernel(self.X_train[i1], self.X_train[i2])
        eta = k11 + k22 - 2 * k12

        alpha2_new_unclipped = alpha2_old + y2 * (E1 - E2) / eta

        if alpha2_new_unclipped > H:
            alpha2_new = H
        elif alpha2_new_unclipped < L:
            alpha2_new = L
        else:
            alpha2_new = alpha2_new_unclipped

        if np.abs(alpha2_new - alpha2_old) < 1e-5:
            return 0
        
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        b1_new = -E1 - y1 * k11 * (alpha1_new - alpha1_old) \
                     - y2 * k12 * (alpha2_new - alpha2_old) + self.b
        b2_new = -E2 - y1 * k12 * (alpha1_new - alpha1_old) \
                     - y2 * k22 * (alpha2_new - alpha2_old) + self.b

        if 0 < alpha1_new < self.C:
            self.b = b1_new
        elif 0 < alpha2_new < self.C:
            self.b = b2_new
        else: # Both new alphas are on bounds
            self.b = (b1_new + b2_new) / 2.0
            
        # Update w for linear kernel
        self._compute_w()

        self._update_error_cache_for_index(i1)
        self._update_error_cache_for_index(i2)
        for k in range(self.n_samples):
             if 0 < self.alphas[k] < self.C and k != i1 and k != i2:
                 self._update_error_cache_for_index(k)
        
        return 1 # Alphas changed

            
    def _update_error_cache_for_index(self, i):
        self.errors[i] = self._decision_function_i(i) - self.y_train[i]            
                

    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    