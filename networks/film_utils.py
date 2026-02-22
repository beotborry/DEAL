import torch
import torch.nn as nn
import torch.nn.functional as F
def get_film_input(demo_feature, input_demo_for_film='aem', age_mean=None, age_std=None):
    (sub_id, age, gender, educat, cdrsb, adas11,
     mmse, apoe4, abeta42, tau, ptau, av45, fdg, group, faq) = demo_feature

    # conert demo_feature to dictionary
    demo_feature = {k: v for k, v in zip(['sub_id', 'age', 'gender', 'educat', 'cdrsb', 'adas11',
                                          'mmse', 'apoe4', 'abeta42', 'tau', 'ptau', 'av45', 'fdg', 'group', 'faq',
                                          ], demo_feature)}

    if 'age_dummy' in input_demo_for_film:
        age_dummy = (group % 2 == 1).float()
        demo_feature['age_dummy'] = age_dummy

    if 'daft' in input_demo_for_film:
        has_apoe4 = torch.ones_like(apoe4)
        has_apoe4[torch.isnan(apoe4)] = 0
        apoe4[torch.isnan(apoe4)] = 0

        has_abeta42 = torch.ones_like(abeta42)
        has_abeta42[torch.isnan(abeta42)] = 0
        abeta42[torch.isnan(abeta42)] = 0

        has_tau = torch.ones_like(tau)
        has_tau[torch.isnan(tau)] = 0
        tau[torch.isnan(tau)] = 0

        has_ptau = torch.ones_like(ptau)
        has_ptau[torch.isnan(ptau)] = 0
        ptau[torch.isnan(ptau)] = 0

        has_av45 = torch.ones_like(av45)
        has_av45[torch.isnan(av45)] = 0
        av45[torch.isnan(av45)] = 0

        has_fdg = torch.ones_like(fdg)
        has_fdg[torch.isnan(fdg)] = 0
        fdg[torch.isnan(fdg)] = 0

        film_input = torch.stack((age, gender, educat, apoe4, has_apoe4, abeta42, has_abeta42,
                                  tau, has_tau, ptau, has_ptau, av45, has_av45, fdg, has_fdg), dim=1)

    else:
        if 'adas11' in input_demo_for_film:
            has_adas11 = torch.ones_like(adas11)
            has_adas11[torch.isnan(adas11)] = 0
            adas11[torch.isnan(adas11)] = 0
            demo_feature.update({'has_adas11': has_adas11, 'adas11': adas11})

        if 'faq' in input_demo_for_film:
            has_faq = torch.ones_like(faq)
            has_faq[torch.isnan(faq)] = 0
            faq[torch.isnan(faq)] = 0
            demo_feature.update({'has_faq': has_faq, 'faq': faq})

        film_input = torch.stack([demo_feature[k] for k in input_demo_for_film], dim=1)

    film_input = film_input.cuda()
    return film_input

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
     'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def __init__(self, film_for_last=False):
        super().__init__()
        self.film_for_last = film_for_last

    def forward(self, x, gammas, betas):
        if not self.film_for_last:
            gammas = gammas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
            betas = betas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return (gammas * x) + betas