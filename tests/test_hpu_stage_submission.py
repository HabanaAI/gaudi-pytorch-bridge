import pytest
import torch
from test_utils import cpu, env_var_in_scope, hpu

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")


class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, hidden_layers=10, threshold=0.35):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.hidden_layers = hidden_layers
        self.fc_in = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_h = torch.nn.ModuleList(
            [torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.hidden_layers)]
        )
        self.fc_out = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        _, _, D = x.shape
        energy = torch.mean(torch.square(x), dim=-1)
        low_energy_frames = torch.nonzero(energy.view(-1) < self.threshold).view(-1)
        x.view((-1, D))[low_energy_frames] *= 0
        x = self.fc_in(x)
        for fc in self.fc_h:
            x = fc(x)
        x = self.fc_out(x)
        return x


@pytest.mark.skip(reason="Tests is chaning env variables")
def test_hpu_lazy_stage_submission():
    with env_var_in_scope({"PT_HPU_MAX_COMPOUND_OP_SIZE": "15", "PT_HPU_ENABLE_STAGE_SUBMISSION": "1"}):
        x = torch.rand(8, 3, 24, device=hpu)
        net = Net(24, 4).to(hpu)
        y = net(x)
        y.to(cpu)


if __name__ == "__main__":
    test_hpu_lazy_stage_submission()
