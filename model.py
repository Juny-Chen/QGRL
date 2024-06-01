from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from core_qnn.quaternion_ops import *
import math
from sklearn.cluster import KMeans
from metrics import cal_clustering_metric


def get_Laplacian_from_weights(weights):
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree


def to_tensor(X):
    if type(X) is torch.Tensor:
        return X
    return torch.Tensor(X)


def get_Laplacian(A):
    device = A.device
    dim = A.shape[0]
    L = A + torch.eye(dim).to(device)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-1 / 2)
    Laplacian = sqrt_D * (sqrt_D * L).t()
    return Laplacian


class QGNNLayer(Module):
    def __init__(self, in_features, out_features, quaternion_ff=True,
                 act=F.relu, init_criterion='he', weight_init='quaternion',
                 seed=None):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.quaternion_ff = quaternion_ff
        self.act = act

        if self.quaternion_ff:
            self.r = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
            self.i = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
            self.j = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
            self.k = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
        else:
            self.commonLinear = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        if self.quaternion_ff:
            winit = {'quaternion': quaternion_init,
                     'unitary': unitary_init}[self.weight_init]
            affect_init(self.r, self.i, self.j, self.k, winit,
                        self.rng, self.init_criterion)

        else:
            stdv = math.sqrt(6.0 / (self.commonLinear.size(0) + self.commonLinear.size(1)))
            self.commonLinear.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if self.quaternion_ff:
            # construct a Hamilton matrix
            r1 = torch.cat([self.r, -self.i, -self.j, -self.k], dim=0)
            i1 = torch.cat([self.i, self.r, -self.k, self.j], dim=0)
            j1 = torch.cat([self.j, self.k, self.r, -self.i], dim=0)
            k1 = torch.cat([self.k, -self.j, self.i, self.r], dim=0)
            cat_kernels_4_quaternion = torch.cat([r1, i1, j1, k1], dim=1)

            # X * W use Hamilton matrix to left product
            mid = torch.mm(x, cat_kernels_4_quaternion)
        else:
            mid = torch.mm(x, self.commonLinear)

        out = torch.mm(adj, mid)

        return self.act(out)


class QGRL(Module):
    def __init__(self,
                 name,
                 X,
                 A,
                 labels,
                 decomposition='symeig',
                 is_norm=True,
                 layers=None,
                 acts=None,
                 max_epoch=10,
                 max_iter=50,
                 learning_rate=10 ** -2,
                 coeff_reg=10 ** -3,
                 seed=114514,
                 lam=-1,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                 ):
        super(QGRL, self).__init__()
        self.name = name
        self.device = device
        self.X = to_tensor(X).to(self.device)
        self.adjacency = to_tensor(A).to(self.device)
        self.labels = to_tensor(labels).to(self.device)

        print(self.device)
        self.decomposition = decomposition
        self.is_norm = is_norm

        self.n_clusters = self.labels.unique().shape[0]
        if layers is None:
            layers = [32, 16]
        self.layers = layers
        self.acts = acts
        assert len(self.acts) == len(self.layers)
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg
        self.seed = seed

        self.data_size = self.X.shape[0]
        self.input_dim = self.X.shape[1]
        print(self.input_dim)

        self.indicator = self.X
        self.embedding = self.X
        self.num_neighbors = 5
        self.links = 0
        self.lam = lam
        self._build_up()

        self.to(self.device)

    def _build_up(self):
        self.pro1 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro2 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro3 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro4 = torch.nn.Linear(self.input_dim, self.layers[0])

        self.qgnn1 = QGNNLayer(self.layers[0] * 4, self.layers[1] * 4, quaternion_ff=True, \
                               act=self.acts[0], init_criterion='he', weight_init='quaternion', seed=self.seed)
        self.qgnn2 = QGNNLayer(self.layers[1] * 4, self.layers[2] * 4, quaternion_ff=True, \
                               act=self.acts[1], init_criterion='he', weight_init='quaternion', seed=self.seed)

    def forward(self, Laplacian):
        x1 = self.pro1(self.X)
        x2 = self.pro2(self.X)
        x3 = self.pro3(self.X)
        x4 = self.pro4(self.X)

        input = torch.cat((x1, x2, x3, x4), dim=1)

        input = self.qgnn1(input, Laplacian)
        input = self.qgnn2(input, Laplacian)
        input = input.reshape(self.data_size, 4, self.layers[2]).sum(dim=1) / 4.
        self.embedding = input

        recons_A = self.embedding.matmul(self.embedding.t())

        return recons_A

    def build_loss_reg(self):
        loss_reg = 0

        for module in self.modules():
            if type(module) is torch.nn.Linear:
                loss_reg += torch.abs(module.weight).sum()
            if type(module) is QGNNLayer:
                loss_reg += (
                            torch.abs(module.r) + torch.abs(module.i) + torch.abs(module.j) + torch.abs(module.k)).sum()

        return loss_reg

    def build_loss(self, recons_A):
        size = self.X.shape[0]

        epsilon = torch.tensor(10 ** -7).to(self.device)
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss_1 = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) \
                 + (1 - self.adjacency).mul((1 / torch.max((1 - recons_A), epsilon)).log())
        loss_1 = loss_1.sum() / (self.data_size ** 2)

        loss_reg = self.build_loss_reg()

        degree = recons_A.sum(dim=1)
        L = torch.diag(degree) - recons_A
        loss_SC = torch.trace(self.embedding.t().matmul(L).matmul(self.embedding)) / (size)

        loss = loss_1 + self.coeff_reg * loss_reg + self.lam * loss_SC
        return loss

    def update_graph(self, embedding):
        weights = embedding.matmul(embedding.t())
        weights = weights.detach()
        return weights

    def clustering(self, weights):
        degree = torch.sum(weights, dim=1).pow(-0.5)
        L = (weights * degree).t() * degree

        if self.decomposition == 'symeig':
            _, vectors = L.symeig(True)
        elif self.decomposition == 'svd':
            vectors, _, __ = torch.svd(L)
        elif self.decomposition == 'eigh':
            _, vectors = torch.linalg.eigh(L)

        indicator = vectors[:, -self.n_clusters:].detach()

        if self.is_norm:
            indicator = indicator / (indicator.norm(dim=1)+10**-10).repeat(self.n_clusters, 1).t()

        indicator = indicator.cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters).fit(indicator)
        prediction = km.predict(indicator)
        acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), prediction)

        return acc, nmi, ari, f1

    def run(self):
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        weights = self.update_graph(self.embedding)
        weights = get_Laplacian_from_weights(weights)

        acc, nmi, ari, f1 = self.clustering(weights)

        best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1

        print('Initial ACC: %.2f, NMI: %.2f, ARI: %.2f' % (acc * 100, nmi * 100, ari * 100))
        objs = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                optimizer.zero_grad()
                recons_A = self(Laplacian)
                loss = self.build_loss(recons_A)
                loss.backward()
                optimizer.step()
                objs.append(loss.item())

            weights = self.update_graph(self.embedding)
            weights = get_Laplacian_from_weights(weights)

            acc, nmi, ari, f1 = self.clustering(weights)
            loss = self.build_loss(recons_A)
            objs.append(loss.item())
            print('{}'.format(epoch) + 'loss: %.4f, ACC: %.2f, NMI: %.2f, ARI: %.2f, F1: %.2f' % (
            loss.item(), acc * 100, nmi * 100, ari * 100, f1 * 100))

            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)

        print("best_acc{},best_nmi{},best_ari{},best_f1{}".format(best_acc, best_nmi, best_ari, best_f1))
        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)
        print(acc_list.mean(), "±", acc_list.std())
        print(nmi_list.mean(), "±", nmi_list.std())
        print(ari_list.mean(), "±", ari_list.std())
        print(f1_list.mean(), "±", f1_list.std())
        return best_acc, best_nmi, best_ari, best_f1

    def build_pretrain_loss(self, recons_A):
        epsilon = torch.tensor(10 ** -7).to(self.device)
        recons_A = recons_A - recons_A.diag().diag()
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) + (1 - self.adjacency).mul(
            (1 / torch.max((1 - recons_A), epsilon)).log())
        loss = loss.sum() / (loss.shape[0] * loss.shape[1])
        loss_reg = self.build_loss_reg()
        loss = loss + self.coeff_reg * loss_reg
        return loss

    def pretrain(self, pretrain_iter, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        print('Start pretraining (totally {} iterations) ......'.format(pretrain_iter))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        for i in range(pretrain_iter):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            loss = self.build_pretrain_loss(recons_A)
            loss.backward()
            optimizer.step()
        print(loss.item())



