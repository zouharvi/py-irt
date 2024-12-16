from py_irt.models import abstract_model
import pyro
import pyro.distributions as dist
import torch

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD

import pyro.contrib.autoguide as autoguide

import pandas as pd

from functools import partial

import numpy as np

import torch.nn as nn
import torch.nn.functional as F



# building off of ProdLDA model for amortization (text only for now) 
# https://pyro.ai/examples/prodlda.html
class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_dimensions, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_dimensions)
        self.fclv = nn.Linear(hidden, num_dimensions)
        # NB: here we set `affine=False` to reduce the number of learning parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # for the effect of this flag in BatchNorm1d
        self.bnmu = nn.BatchNorm1d(num_dimensions, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_dimensions, affine=False)  # to avoid component collapse

    def forward(self, inputs):

        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale

@abstract_model.IrtModel.register("amortized_4pl_score")
class AmortizedFourParamLogScore(abstract_model.IrtModel):
    def __init__(
        self, *, 
        priors: str, 
        num_items: int, 
        num_subjects: int, 
        verbose: bool = False, 
        device: str = "cpu",
        vocab_size: int,
        dropout: float,
        hidden: int,
        **kwargs
    ):
        super().__init__(
            device=device, num_items=num_items, num_subjects=num_subjects, verbose=verbose
        )
        # initialize the class with all arguments provided to the constructor
        self.num_dimensions = 1
        self.device = torch.device(device)
        self.drop = dropout
        self.hidden = hidden

        self.vocab_size = vocab_size

        self.encoder_diff = Encoder(vocab_size, self.num_dimensions, self.hidden, self.drop).to(self.device)
        self.encoder_disc = Encoder(vocab_size, self.num_dimensions, self.hidden, self.drop).to(self.device)

    def model_irt(self, models, items, obs):
        options_cpu = dict(dtype=torch.float, device="cpu")
        models = models.to(self.device)
        items = items.to(self.device)
        obs = obs.to("cpu")

        with pyro.plate("thetas"):
            ability = pyro.sample('theta', dist.Normal(
                torch.zeros(self.num_subjects, **options_cpu),
                torch.ones(self.num_subjects, **options_cpu)
            )).to(self.device)

        with pyro.plate("items", len(items)):
            # sample the item difficulty from the prior distribution
            diff_prior_loc = torch.zeros(len(items), **options_cpu).unsqueeze(1).float()
            diff_prior_scale = torch.ones(len(items), **options_cpu).fill_(1.e3).unsqueeze(1).float()
            diff = pyro.sample('diff', dist.Normal(diff_prior_loc, diff_prior_scale).to_event(1)).to(self.device)

            # sample the item discriminability from the prior distribution
            disc_prior_loc = torch.zeros(len(items), **options_cpu).unsqueeze(1).float()
            disc_prior_scale = torch.ones(len(items), **options_cpu).fill_(1.e3).unsqueeze(1).float()
            disc = pyro.sample('disc', dist.Normal(disc_prior_loc, disc_prior_scale).to_event(1)).to(self.device)

        scale_obs = pyro.sample(
            'scale_obs',
            dist.Gamma(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device)
            )
        )

        with pyro.plate("observe_data", len(obs)):
            p_star = torch.sigmoid(disc * (ability[models] - diff)).to("cpu")
            pyro.sample('obs', dist.Normal(loc=p_star, scale=1.0/scale_obs).to_event(1), obs=obs)
        
    def guide_irt(self, models, items, obs):
        options = dict(dtype=torch.float, device=self.device)
        options_cpu = dict(dtype=torch.float, device="cpu")
        # vectorize
        items = items.to(self.device)

        # register learnable params in the param store
        with pyro.plate("systems"):
            m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_subjects, **options))
            s_theta_param = pyro.param(
                "scale_ability",
                torch.ones(self.num_subjects, **options),
                constraint=constraints.positive,
            )
            dist_theta = dist.Normal(m_theta_param.to("cpu"), s_theta_param.to("cpu"))
            pyro.sample("theta", dist_theta)

        with pyro.plate("items", len(items)):
            loc_diffs_all, scale_diffs_all = self.encoder_diff.forward(items)
            loc_discs_all, scale_discs_all = self.encoder_disc.forward(items)
            
            dist_diff = dist.Normal(loc_diffs_all.to("cpu"), scale_diffs_all.to("cpu"))
            pyro.sample('diff', dist_diff.to_event(1))

            dist_disc = dist.Normal(loc_discs_all.to("cpu"), scale_discs_all.to("cpu"))
            pyro.sample('disc', dist_disc.to_event(1))

        # sample statements
        alpha_obs_param = pyro.param(
            "alpha_obs",
            torch.tensor(1.0, device="cpu"),
            constraint=constraints.positive,
        )
        beta_obs_param = pyro.param(
            "beta_obs",
            torch.tensor(1.0, device="cpu"),
            constraint=constraints.positive,
        )
        scale_obs = pyro.sample("scale_obs", dist.Gamma(alpha_obs_param, beta_obs_param))


    def get_model(self):
        return self.model_irt

    def get_guide(self):
        return self.guide_irt

    def fit(self, models, items, responses, num_epochs):
        """Fit the IRT model with variational inference"""
        # need to step with IRT loss and with reconstruction loss

        optim = Adam({"lr": 0.1})
        svi = SVI(self.model_irt, self.guide_irt, optim, loss=Trace_ELBO())
        #svi_diff = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(models, items, responses)
            #loss_diff = svi_diff.step(items)
            if j % 100 == 0 and self.verbose:
                print("[epoch %04d] irt loss: %.4f" % (j + 1, loss))
                #print("[epoch %04d] recon loss: %.4f" % (j + 1, loss_diff))

        print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        #print("[epoch %04d] loss: %.4f" % (j + 1, loss_diff))
        values = ["loc_diff", "scale_diff", "loc_ability", "scale_ability"]

    def export(self, items):
        items = torch.tensor(items, dtype=torch.float)
        diffs, _ = self.encoder_diff.forward(items.to(self.device))
        diffs = diffs.squeeze().cpu().detach().numpy()
        discs, _ = self.encoder_disc.forward(items.to(self.device))
        discs = discs.squeeze().cpu().detach().numpy()

        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": diffs.tolist(),
            "disc": discs.tolist(),
        }

    def fit_MCMC(self, models, items, responses, num_epochs):
        """Fit the IRT model with MCMC"""
        sites = ["theta", "b"]
        nuts_kernel = NUTS(self.model_vague, adapt_step_size=True)
        hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100).run(
            models, items, responses
        )
        theta_sum = self.summary(hmc_posterior, ["theta"]).items()
        b_sum = self.summary(hmc_posterior, ["b"]).items()
        
    def predict(self, subjects, items, params_from_file=None):
        """predict p(correct | params) for a specified list of model, item pairs"""
        if params_from_file is not None:
            model_params = params_from_file
        else:
            model_params = self.export(items)
        abilities = np.array([model_params["ability"][i] for i in subjects])
        diffs = np.array(model_params["diff"])
        discs = np.array(model_params["disc"])
        return 1 / (1 + np.exp(-discs*(abilities - diffs)))

    def summary(self, traces, sites):
        """Aggregate marginals for MCM"""
        marginal = (
            EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        )
        print(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1)[
                ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
        return site_stats
