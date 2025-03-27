import torch
import matplotlib.pyplot as plt

class Block:
    def __init__(self, px, py, device="cpu"):
        self.px = px
        self.py = py
        self.device = device
        self.mask = None

    def get_mask(self):
        return



class DiscBlock(Block):
    def __init__(self, px, py, r, device="cpu"):
        assert r > 0, 'Disc Error'
        super().__init__(px, py, device=device)
        self.c = torch.tensor([px + r, py + r], device=device)
        self.r = torch.tensor([r], dtype=torch.float32, device=device)
        self.w = 2 * r + 1
        self.h = self.w

    def get_mask(self):
        self.mask = self.create_mask(self.r, device=self.device)


    def create_mask(self, r, device):
        size = 2 * r + 1
        y, x = torch.meshgrid(
            torch.arange(size, device=device), 
            torch.arange(size, device=device), 
            indexing="ij")
        c = r
        mask = (x - c) ** 2 + (y - c) ** 2 <= r ** 2
        return mask.to(device)
    

    def get_depth(self, start, end):
        n, v = end - start, self.c - start
        n_2 = torch.dot(n, n).unsqueeze(0)
        t_proj = torch.dot(v, n).unsqueeze(0) / n_2
        t = t_proj * n
        d = torch.norm(v - t, keepdim=True)
        l = torch.norm(t, keepdim=True)
        h = self.r**2 - d**2
        if h < 0 or l**2 < n_2:
            return torch.tensor([-1.], device=self.device)
        return l - torch.sqrt(h)


class TriangleBlock(Block):
    def __init__(self, vertices, device="cpu"):
        assert len(vertices) == 3, 'Triangle Error'
        self.vertices = [
            torch.tensor(vertices[0], device=device),
            torch.tensor(vertices[1], device=device),
            torch.tensor(vertices[2], device=device)
        ]
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        super().__init__(min_x, min_y, device=device)
        self.w, self.h = torch.tensor([max_x - min_x + 1], device=device), torch.tensor([max_y - min_y + 1], device=device)
    
    def get_mask(self):
        self.mask = self.create_mask(self.w, self.h, self.vertices, device=self.device)


    def create_mask(self, w, h, vertices, device):
        y_range = torch.arange(h, device=device)
        x_range = torch.arange(w, device=device)
        yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        def edge_function(v_0, v_1, p):
            return (p[..., 0] - v_0[0]) * (v_1[1] - v_0[1]) - (p[..., 1] - v_0[1]) * (v_1[0] - v_0[0])
        
        v0, v1, v2 = vertices[0], vertices[1], vertices[2]
        area = edge_function(v0, v1, v2)
        w0 = edge_function(v1, v2, points) / area
        w1 = edge_function(v2, v0, points) / area
        w2 = edge_function(v0, v1, points) / area

        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        return mask.reshape(h, w)


    def get_depth(self, start, end):
        start = torch.tensor(start, device=self.device)
        end = torch.tensor(end, device=self.device)
        n = end - start
        n_norm = torch.norm(n, keepdim=True)
        def edge_depth(v0, v1):
            e = v1 - v0
            b = v1 - start
            A = torch.stack([n, e], dim=1)
            if torch.abs(torch.det(A)) < 1e-6:
                return
            s = torch.linalg.solve(A, b)
            return s

        ts = set([])
        for i in range(3):
            s = edge_depth(self.vertices[i], self.vertices[(i+1)%3])
            if s[0] >= 0 and 0 <= s[1] <= 1:
                if s[0] < 1:
                    return torch.tensor([-1.], device=self.device)
                ts.add(s[0])
        
        if len(ts) == 1:
            return torch.tensor([-1.], device=self.device)
        return min(ts) * n_norm

if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = [
        [3., 0.], 
        [0., 3.], 
        [3., 3.], 
    ]
    triangle = TriangleBlock(v, device="cpu")
    a = triangle.get_depth(
        start=[0., 0.],
        end=[1., 1.],
    )
    print(a)

    pass