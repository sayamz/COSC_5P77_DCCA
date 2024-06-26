import torch 
#%%
def DCCA_Loss(H_list):
    
    r = 1e-4
    eps = 1e-8

    top_k = 10

    AT_list =  []

    for H in H_list:
        assert torch.isnan(H).sum().item() == 0 

        o_shape = H.size(0)  # N
        m = H.size(1)   # out_dim

        Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(Hbar).sum().item() == 0

        A, S, B = Hbar.svd(some=True, compute_uv=True)

        A = A[:, :top_k]

        assert torch.isnan(A).sum().item() == 0

        S_thin = S[:top_k]

        S2_inv = 1. / (torch.mul( S_thin, S_thin ) + eps)

        assert torch.isnan(S2_inv).sum().item() == 0

        T2 = torch.mul( torch.mul( S_thin, S2_inv ), S_thin )

        assert torch.isnan(T2).sum().item() == 0

        T2 = torch.where(T2>eps, T2, (torch.ones(T2.shape)*eps).to(H.device).double())


        T = torch.diag(torch.sqrt(T2))

        assert torch.isnan(T).sum().item() == 0

        T_unnorm = torch.diag( S_thin + eps )

        assert torch.isnan(T_unnorm).sum().item() == 0

        AT = torch.mm(A, T)
        AT_list.append(AT)

    M_tilde = torch.cat(AT_list, dim=1)

    assert torch.isnan(M_tilde).sum().item() == 0

    _, S, _ = M_tilde.svd(some=True)

    assert torch.isnan(S).sum().item() == 0
    use_all_singular_values = False
    if not use_all_singular_values:
        S = S[:top_k]

    corr = torch.sum(S )
    assert torch.isnan(corr).item() == 0

    loss = - corr
    return loss