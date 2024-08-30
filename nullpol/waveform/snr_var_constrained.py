import jax.numpy as jnp
from jax import jacrev

def arctan(x, y):
    output = jnp.arctan(y/x)
    if x >= 0 and y <= 0:
        return output + 2 * np.pi
    elif x <=0 and y <= 0:
        return output + np.pi
    elif x <= 0 and y >= 0:
        return output + np.pi
    return output

def compute_eigen_of_H(antenna_pattern):
    H = jnp.conj(antenna_pattern.T) @ antenna_pattern
    eigvals, eigvecs = jnp.linalg.eig(H)
    idx = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]    
    return eigvals, eigvecs

# def compute_antenna_pattern_num_derv(ra, dec, psi, gps_time, dets, delta=1e-3):
#     F_ra_plus = compute_antenna_pattern(ra+delta, dec, psi, gps_time, dets)
#     F_ra_minus = compute_antenna_pattern(ra-delta, dec, psi, gps_time, dets)
#     F_ra = (F_ra_plus - F_ra_minus) / (delta * 2)
#     F_dec_plus = compute_antenna_pattern(ra, dec+delta, psi, gps_time, dets)
#     F_dec_minus = compute_antenna_pattern(ra, dec-delta, psi, gps_time, dets)
#     F_dec = (F_dec_plus - F_dec_minus) / (delta * 2)
#     return F_ra, F_dec

def compute_sky_var_constrained_waveform_at_one_bin_helper(whitened_strain,
                                                           whitened_antenna_pattern,
                                                           whitened_antenna_pattern_ra,
                                                           whitened_antenna_pattern_dec,
                                                           alpha=0.):
    H_eigvals, H_eigvecs = compute_eigen_of_H(whitened_antenna_pattern)
    H_eigvals_0 = H_eigvals[0]
    H_eigvals_1 = H_eigvals[1]
    H_eigvecs_0 = H_eigvecs[:,0]
    H_eigvecs_1 = H_eigvecs[:,1]
    # Obtain the q basis
    q_0 = (whitened_antenna_pattern @ H_eigvecs_0) / jnp.sqrt(jnp.abs(H_eigvals_0))
    q_1 = (whitened_antenna_pattern @ H_eigvecs_1) / jnp.sqrt(jnp.abs(H_eigvals_1))
    # Compute the matrix M
    whitened_antenna_pattern_dec_H_eigvecs_0 = whitened_antenna_pattern_dec @ H_eigvecs_0
    whitened_antenna_pattern_ra_H_eigvecs_0 =  whitened_antenna_pattern_ra @ H_eigvecs_0
    whitened_antenna_pattern_dec_H_eigvecs_1 = whitened_antenna_pattern_dec @ H_eigvecs_1
    whitened_antenna_pattern_ra_H_eigvecs_1 =  whitened_antenna_pattern_ra @ H_eigvecs_1
    M_A = (jnp.sum(jnp.abs(whitened_antenna_pattern_dec_H_eigvecs_0)**2) + jnp.sum(jnp.abs(whitened_antenna_pattern_ra_H_eigvecs_0)**2)) / jnp.abs(H_eigvals_0)
    M_B = (jnp.sum(jnp.abs(whitened_antenna_pattern_dec_H_eigvecs_1)**2) + jnp.sum(jnp.abs(whitened_antenna_pattern_ra_H_eigvecs_1)**2)) / jnp.abs(H_eigvals_1)
    M_C = (jnp.dot(whitened_antenna_pattern_dec_H_eigvecs_0, whitened_antenna_pattern_dec_H_eigvecs_1) + jnp.dot(whitened_antenna_pattern_ra_H_eigvecs_0, whitened_antenna_pattern_ra_H_eigvecs_1)) / jnp.sqrt(jnp.abs(H_eigvals_0 * H_eigvals_1))
    M = jnp.array([[M_A, M_C], [M_C, M_B]])
    M_eigvals, M_eigvecs = jnp.linalg.eig(M)
    idx = jnp.argsort(M_eigvals)[::-1]
    M_eigvals = M_eigvals[idx]
    M_eigvecs = M_eigvecs[:,idx]
    M_eigvals_0 = M_eigvals[0]
    M_eigvals_1 = M_eigvals[1]
    M_eigvecs_0 = M_eigvecs[:,0]
    M_eigvecs_1 = M_eigvecs[:,1]
    X_q = jnp.array([jnp.dot(whitened_strain, q_0),
                    jnp.dot(whitened_strain, q_1)])
    alpha_x = jnp.dot(X_q, M_eigvecs_0)
    beta_x =  jnp.dot(X_q, M_eigvecs_1)
    l_x = jnp.sqrt(alpha_x ** 2 + beta_x ** 2)
    phi_x = arctan(alpha_x, beta_x)
    psi_hat = arctan((jnp.cos(2 * phi_x) - 2 * alpha * (M_eigvals_0 - M_eigvals_1) / l_x**2), jnp.sin(2 * phi_x)) / 2
    rho_hat = l_x * jnp.cos(psi_hat - phi_x)
    U = jnp.array([q_0, q_1]).T
    Q = jnp.array([M_eigvecs_0, M_eigvecs_1])
    output = rho_hat * (U @ Q @ jnp.array([jnp.cos(psi_hat), jnp.sin(psi_hat)]))
    return output

def compute_sky_var_constrained_waveform_at_one_bin_helper_jacobian_matrix(whitened_strain,
                                                                           whitened_antenna_pattern,
                                                                           whitened_antenna_pattern_ra,
                                                                           whitened_antenna_pattern_dec,
                                                                           alpha=0.):
    partial_f = lambda x: x - compute_sky_var_constrained_waveform_at_one_bin_helper(x,
                                                                                     whitened_antenna_pattern,
                                                                                     whitened_antenna_pattern_ra,
                                                                                     whitened_antenna_pattern_dec,
                                                                                     alpha)
    jacobian = jacrev(partial_f, holomorphic=True)(whitened_strain)
    return jacobian