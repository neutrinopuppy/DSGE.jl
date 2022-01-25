############ THE FOLLOWING ARE THE DIRECT TRANSLATIONS OF THE PYTHON CODE


# behold, the solve method (this is gen_sys, because the universe hates us)
function solve_ocb(m::AbstractDSGEModel; l_max::Int32 = 3, k_max::Int32 = 17, parallel::Bool = false, verbose::Symbol = :none)
    Γ0, Γ1, C, Ψ, Π = eqcond(m)

    # set default values of l_max
    if l_max < 2 and k_max > 0
        @warn " l_max must be at least 2 (is $l_max). Correcting..."
        l_max = 2
    end
    l_max += 1

    m <= Setting(:lks, [l_max, k_max])

    # TODO what is thiiiiis
    # start
    vv0 = get_setting(m, :vv)

    # z-space is the space of the original variables
    dimx = length(vv0)
    dimeps = length(m.shocks)

    # y-space is the space of the original variables augmented by the shocks
    c_arg = list(vv0)[get_setting(m, :const_var)]
    fc0 = -fc0/fb0[c_arg]
    fb0 = -fb0/fb0[c_arg]

    # create auxiliry vars for those both in A & C
    inall = ~fast0(AA0, 0) & ~fast0(CC0, 0)
    if np.any(inall)
        vv0 = np.hstack((vv0, [v + '_lag' for v in vv0[inall]]))
        AA0 = np.pad(AA0, ((0, sum(inall)), (0, sum(inall))))
        BB0 = np.pad(BB0, ((0, sum(inall)), (0, sum(inall))))
        CC0 = np.pad(CC0, ((0, sum(inall)), (0, sum(inall))))
        DD0 = np.pad(DD0, ((0, sum(inall)), (0, 0)))
        fb0 = np.pad(fb0, (0, sum(inall)))
        fc0 = np.pad(fc0, (0, sum(inall)))

        if ZZ0 != nothing
            ZZ0 = np.pad(ZZ0, ((0, 0), (0, sum(inall))))
        end
        BB0[-sum(inall):, -sum(inall):] = np.eye(sum(inall))
        BB0[-sum(inall):, :-sum(inall)][:, inall] = -np.eye(sum(inall))
        CC0[:, -sum(inall):] = CC0[:, :-sum(inall)][:, inall]
        CC0[:, :-sum(inall)][:, i  nall] = 0
    end
    # create representation in y-space
    A0 = np.pad(AA0, ((0, dimeps), (0, dimeps)))
    BB0 = sl.block_diag(BB0, np.eye(dimeps))
    CC0 = np.block([[CC0, DD0], [np.zeros((dimeps, AA0.shape[1]))]])
    fb0 = np.pad(fb0, (0, dimeps))
    if fd0 != nothing
        fc0 = -np.hstack((fc0, fd0))
    else
        fc0 = np.pad(fc0, (0, dimeps))
    end
    inq = ~fast0(CC0, 0) | ~fast0(fc0)
    inp = (~fast0(AA0, 0) | ~fast0(BB0, 0)) & ~inq

    # check dimensionality
    dimq = sum(inq)
    dimp = sum(inp)
    #
    #             # create hx. Do this early so that the procedure can be stopped if get_hx_only
    if ZZ0 != nothing
        # must create dummies
        zp = np.empty(dimp)
        zq = np.empty(dimq)
        zc = np.empty(1)
    else
        zp = ZZ0[:, inp[:-dimeps]]
        zq = ZZ0[:, inq[:-dimeps]]
        zc = ZZ1
    end

    AA = np.pad(AA0, ((0, 1), (0, 0)))
    BBU = np.vstack((BB0, fb0))
    CCU = np.vstack((CC0, fc0))
    BBR = np.pad(BB0, ((0, 1), (0, 0)))
    CCR = np.pad(CC0, ((0, 1), (0, 0)))
    BBR[-1, list(vv0).index(str(self.const_var))] = -1

    fb0[list(vv0).index(str(self.const_var))] = 0

    self.svv = vv0[inq[:-dimeps]]
    self.cvv = vv0[inp[:-dimeps]]
    self.vv = np.hstack((self.cvv, self.svv))

    self.dimx = len(self.vv)
    self.dimq = dimq
    self.dimp = dimp
    self.dimy = dimp+dimq
    self.dimeps = dimeps
    self.hx = zp, zq, zc

    if get_hx_only
        return self
    end

    PU = -np.hstack((BBU[:, inq], AA[:, inp]))
    MU = np.hstack((CCU[:, inq], BBU[:, inp]))

    PR = -np.hstack((BBR[:, inq], AA[:, inp]))
    MR = np.hstack((CCR[:, inq], BBR[:, inp]))
    gg = np.pad([float(self.x_bar)], (dimp+dimq-1, 0))

    # avoid QL in jitted funcs
    R, Q = sl.rq(MU.T)
    MU = R.T
    PU = Q @ aca(PU)

    R, Q = sl.rq(MR.T)
    MR = R.T
    PR = Q @ aca(PR)
    gg = Q @ gg

    if solution == nothing
        solution = :klein
    end

    if isinstance(solution, str)
        if solution == :speed_kills
            omg, lam = speed_kills(PU, MU, dimp, dimq, tol=1e-4)
        else
            omg, lam = klein(PU, MU, nstates=dimq, verbose=verbose, force=False)
        end
    else
        omg, lam = solution
    end


    # finally add relevant stuff to the class

    fq0 = fc0[inq]
    fp1 = fb0[inp]
    fq1 = fb0[inq]

    self.sys = omg, lam, self.x_bar
    self.ff = fq1, fp1, fq0

    # preprocess all system matrices until (l_max, k_max)
    #
    preprocess(self, PU, MU, PR, MR, gg, fq1, fp1, fq0, parallel, verbose)

    if verbose == :full
        print('[get_sys:]'.ljust(15, ' ')+' Creation of system matrices finished in %ss.' %
              np.round(time.time() - st, 3))
    end
    return m
end





PU, MU, PR, MR, gg, fq1, fp1, fq0, omg, lam, x_bar, l_max, k_max

function preprocess_jittable(P_hat::AbstractMatrix{S}, M_hat::AbstractMatrix{S}, P::AbstractMatrix{S}, M::AbstractMatrix{S}, h, fq1, fp1, fq0, TTT::AbstractMatrix{S}, RRR::AbstractMatrix{S}, x_bar::Float64, l_max::Int64, k_max::Int64) where S <: Real
    p, q = size(TTT)
    l_max += 1
    k_max += 1

    M_hat_22 = M_hat_22[p:end, q:end]
    if norm(M_hat_22) > 1/ϵ # NOTE: in the original code, this epsilon is a model setting
        @warn "At least one control intedermined."
    end


    M_hat_22i = inv(M_hat_22)
    M_hat[q:end] = M_hat_22i * M_hat[q:end]
    P_hat[q:end] = M_hat_22i * P_hat[q:end]

    M_22 = M[q:end, q:end]
    M_22i = inv(M_22)
    M[q:end] = M22i * M[q:end]
    P[q:end] = M22i * P[q:end]
    h[q:end] = M22i * h[q:end]

    pmat  = Array{Float64}(undef, l_max, k_max, p, q)
    qmat  = Array{Float64}(undef, l_max, k_max, p, q)
    pterm = Array{Float64}(undef, l_max, k_max, p)
    qterm = Array{Float64}(undef, l_max, k_max, p)

    pmat[0,0] = omg
    pterm[0,0] = zeros(p)
    qmat[0,0] = lam
    qterm[0,0] = zeros(q)

    for l in 0:l_max
        for k in 0:k_max
            if k || l
                l_last = max(l-1,0)
                k_last = l ? k : max(k-1, 0)

                qmat[l,k], qterm[l,k] = get_lam(pmat[l_last, k_last],
                                                pterm[l_last, k_last],
                                                P_hat, M_hat, P, M, h, l)

                pmat[l,k], pterm[l,k] = get_omg(pmat[l_last, k_last],
                                                pterm[l_last, k_last],
                                                qmat[l,k], qterm[l,k],
                                                P_hat, M_hat, P, M, h, l)
            end
        end
    end

    bmat = Array{Float64}(undef, 5, l_max, k_max, q)
    bterm = Array{Float64}(undef, 5, l_max, k_max)


    for l in 0:l_max
        @distributed for k in 0:k_max
            lam = I(q)
            xi = zeros(q)

            for s in 0:l+k+1
                l_loc = max(l-s, 0)
                k_loc = max(min(k, k+l-s), 0)

                y2r = fp1 * pmat[l_loc, k_loc] + fq1 * qmat[l_loc, k_loc] + fq0
                cr = fp1 * pterm[l_loc, k_loc] + fq1 * qterm[l_loc, k_loc]

                if s == 0
                    bmat[0, l, k] = y2r * lam
                    bterm[0, l, k] = cr + y2r * xi
                elseif s == l-1
                    bmat[1, l, k] = y2r * lam
                    bterm[1, l, k] = cr + y2r * xi
                elseif s == l
                    bmat[2, l, k] = y2r * lam
                    bterm[2, l, k] = cr + y2r * xi
                elseif s == l+k-1
                    bmat[3, l, k] = y2r * lam
                    bterm[3, l, k] = cr + y2r * xi
                elseif s == l+k
                    bmat[4, l, k] = y2r * lam
                    bterm[4, l, k] = cr + y2r * xi

                    lam = qmat[l_loc, k_loc] * lam
                    xi = qmat[l_loc, k_loc] * xi + qterm[l_loc, k_loc]
                end
            end
        end
    end

    return pmat, qmat, pterm, qterm, bmat, bterm
end



function get_lam(Ω, ψ, P_hat, M_hat, P, M, h, l)
    dimp, dimq = size(Ω)

    A = l ? P_hat : P
    B = l ? M_hat : M
    c = l ? zeros(dimq) : h[1:dimq]

    # original code uses contiguous arrays here
    inv = inv(A[1:dimq, 1:dimq] + A[1:dimq, dimq:end] * Ω)
    λ = inv * B[1:dimq, 1:dimq]
    xi = inv * (c - A[1:dimq, dimq:end]) * ψ

    return λ, xi
    end

    function get_omg(Ω, ψ, λ, xi, P_hat, M_hat, P, M, h, l)
        dimp, dimq = size(Ω)

        A = l ? P_hat : P
        B = l ? M_hat : M
        c = l ? zeros(dimp) : h[dimq:end]

        dum = A[dimq:end. 1:dimq] + A[dimq:end, dimq:end] * Ω
        ψ = dum * xi + A[dimq:end, dimq:end] * ψ - c
        Ω = dum * λ - B[dimq:end, 1:dimq]

        return Ω, ψ
    end


    function
