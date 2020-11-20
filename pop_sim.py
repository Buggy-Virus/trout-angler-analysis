import numpy as np

class pop_sim:
    def __init__(self, f, s, n, a, b, angler_rule=0):
        self.t = 0
        self.f = f
        self.s = s
        self.s_list = []
        self.n = n
        self.n_list = [n]
        self.a = a
        self.b = b
        self.growth_list = []

        if angler_rule > 2:
            raise Exception(f"Not a valid angler_rule: {angler_rule}")
        self.angler_rule = angler_rule

    def angler_exact_limit(self, s, n, b, a):
        s_new = np.copy(s)
        s_new[b] = (n[b] * s[b] - a) / n[b]

        return s_new

    def angler_lower_limit(self, s, n, b, a):
        s_new = np.copy(s)
        affected_total = np.sum(n[b:])

        for i in range(b, s.size):           
            s_new[i] = (n[i][0] * s[i] - a * n[i][0] / affected_total) / n[i][0]

        return s_new
        
    def apply_angler_effect(self, s, n, b, a):
        if self.angler_rule == 0:
            return s
        elif self.angler_rule == 1:
            return self.angler_exact_limit(s, n, b, a)
        elif self.angler_rule == 2:
            return self.angler_lower_limit(s, n, b, a)

    def step_forward(self):
        self.t += 1

        effective_s = self.apply_angler_effect(self.s, self.n, self.b, self.a)
        self.s_list.append(effective_s)

        cur_leslie = np.diag(effective_s)
        cur_leslie = np.append(cur_leslie, np.zeros((np.size(cur_leslie,0), 1)), axis=1)
        cur_leslie = np.append(self.f.reshape(1, self.f.size), cur_leslie, axis = 0)

        eigenvalues, _ = np.linalg.eig(cur_leslie)
        growth = eigenvalues[np.argmax(np.absolute(eigenvalues))]
        self.growth_list.append(growth)

        new_n = np.matmul(cur_leslie, self.n)
        self.n_list.append(new_n)
        self.n = new_n

    def step_forward_by(self, steps):
        print(f"Incrementing time by {steps} steps")

        for i in range(steps):
            if (i % 100 == 0 and i != 0):
                print(f"Processed {i} time steps")
            self.step_forward()