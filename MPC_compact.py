# =============================================================================
# MODEL PREDICTIVE CONTROLLER FOR A ROTARY INVERTED PENDULUM
# =============================================================================
# This script contains the calculations and simulations for the MPC controller 
# design report.
#
# Script Structure:
# -----------------
# 1. Imports              : External libraries and configuration.
# 2. System Matrices      : Discrete-time system definitions.
# 3. Helper Functions     : Constraints, predictions, LQR, and terminal sets.
# 4. Simulation Functions : Closed-loop MPC and LQR simulation routines.
# 5. Study Functions      : Specific experiments and plots for the report.
# 6. Entry Point          : Main execution block.
#
# Usage:
# ------
# Navigate to Section 6 (Entry Point) at the bottom of the file and
# uncomment the specific study function(s) you wish to execute.
#
# Authors: Pieter de Vries Robbé & Mark van Gelder
# =============================================================================

# =============================================================================
# 1. IMPORTS
# =============================================================================

import time
import numpy as np
import scipy.linalg as la
import scipy.signal as signal
from scipy.optimize import minimize, linprog 
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

# Set random seed
np.random.seed(26) 

# =============================================================================
# 2. SYSTEM MATRICES
# =============================================================================

def get_system_matrices(discrete_time_study=False):
    '''
    Computes state-space matrices for the rotary inverted pendulum.

    Parameters:
            discrete_time_study (bool): If True, prints discretization analysis.

    Returns:
            A (np.ndarray): Discrete-time state matrix.
            B (np.ndarray): Discrete-time input matrix.
            Ts (float): The sampling time.
    '''
    # Physical parameters of the rotary inverted pendulum
    m_a, m_p, l_a, c_p = 0.058, 0.023, 0.162, 0.08
    J_a, J_p, g        = 5.073e-4, 1.96e-4, 9.81
    R_a, k_i, k_b      = 1.225, 0.0175, 0.0175

    # Intermediate terms for simplified matrix construction
    P1 = J_a + m_p * l_a**2
    P3 = m_p * l_a * c_p
    P4 = J_p + m_p * c_p**2
    P5 = m_p * c_p * g
    P6 = k_b * k_i / R_a
    P7 = k_i / R_a
    DL = P1*P4 - P3**2

    # Continuous-time state-space matrices (Ac, Bc)
    Ac = np.array([[0,0,1,0],[0,0,0,1],
                   [0, P3*P5/DL, -P4*P6/DL, 0],
                   [0, P1*P5/DL, -P3*P6/DL, 0]])
    Bc = np.array([[0],[0],[P4*P7/DL],[-P3*P7/DL]])

    # Calculate sampling time based on the fastest dynamics (rule of thumb)
    eigs   = la.eigvals(Ac)
    rp     = np.abs(np.real(eigs))
    Ts     = round(1.0 / np.max(rp[rp > 1e-5]) / 15.0, 3)

    # Discretize the system using zero-order hold
    sys_d  = signal.StateSpace(Ac, Bc, np.eye(4), np.zeros((4,1))).to_discrete(Ts)

    if discrete_time_study:
        print('\n')
        print('=========Discrete time system study=========')
        print('Sampling time calculation:')
        print(f'Matrix A_c: \n {Ac}')
        print(f'Maximum real part of eigenvalue: {np.max(rp[rp > 1e-5])}')
        print(f'Fastest dynamics: {1.0 / np.max(rp[rp > 1e-5])}')
        print(f'Fastest dynamics / 15 = {1.0 / np.max(rp[rp > 1e-5]) / 15.0}')
        print(f'Sampling time: {Ts}')
        print('\n')
        print('Discrete time matrices:')
        print(f'Matrix A_d: \n {sys_d.A}')
        print(f'Matrix B_d: \n {sys_d.B}')
        print('\n')

    return sys_d.A, sys_d.B, Ts


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def _constraints():
    '''
    Defines the state and input constraints for the rotary pendulum.

    Returns:
            Fu (np.ndarray): Input constraint matrix.
            fu (np.ndarray): Input constraint vector.
            Fx (np.ndarray): State constraint matrix.
            fx (np.ndarray): State constraint vector.
    '''
    Fu = np.array([[1.],[-1.]])
    fu = np.array([10., 10.])
    Fx = np.array([[1,0,0,0],[-1,0,0,0],[0,1,0,0],[0,-1,0,0],[0,0,1,0],[0,0,-1,0],[0,0,0,1],[0,0,0,-1]], dtype=float)
    fx = np.array([0.5*np.pi, 0.5*np.pi, 0.3, 0.3, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
    return Fu, fu, Fx, fx


def _prediction_matrices(A, B, N):
    '''
    Builds the stacked prediction matrices T and S for the MPC problem.

    Parameters:
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            N (int): Prediction horizon.

    Returns:
            T (np.ndarray): Stacked matrix for state prediction (nN x n).
            S (np.ndarray): Stacked matrix for input effect (nN x mN).
    '''
    n, m = A.shape[0], B.shape[1]
    T = np.zeros((n*N, n))
    S = np.zeros((n*N, m*N))
    for i in range(1, N+1):
        T[(i-1)*n:i*n, :] = np.linalg.matrix_power(A, i)
        for j in range(1, i+1):
            S[(i-1)*n:i*n, (j-1)*m:j*m] = np.linalg.matrix_power(A, i-j) @ B
    return T, S


def _qp_cost(A, B, Q, R, N):
    '''
    Computes the Hessian, Qbar, and terminal cost matrix P for the MPC QP.

    Parameters:
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            Q (np.ndarray): State weighting matrix.
            R (float): Input weighting scalar.
            N (int): Prediction horizon.

    Returns:
            H (np.ndarray): Hessian matrix for the QP.
            Qbar (np.ndarray): Stacked state weighting matrix.
            P (np.ndarray): Terminal cost matrix from DARE.
    '''
    P    = la.solve_discrete_are(A, B, Q, np.array([[R]]))
    _, S = _prediction_matrices(A, B, N)
    Qbar = la.block_diag(np.kron(np.eye(N-1), Q), P)
    Rbar = np.kron(np.eye(N), np.array([[R]]))
    H    = 2*(S.T @ Qbar @ S + Rbar)
    H    = (H + H.T)/2
    return H, Qbar, P


def _stack_constraints(Fu, fu, Fx, fx, N):
    '''
    Stacks single-step constraints over the prediction horizon N.

    Parameters:
            Fu (np.ndarray): Single-step input constraint matrix.
            fu (np.ndarray): Single-step input constraint vector.
            Fx (np.ndarray): Single-step state constraint matrix.
            fx (np.ndarray): Single-step state constraint vector.
            N (int): Prediction horizon.

    Returns:
            Fu_bar (np.ndarray): Stacked input constraint matrix.
            fu_bar (np.ndarray): Stacked input constraint vector.
            Fx_bar (np.ndarray): Stacked state constraint matrix.
            fx_bar (np.ndarray): Stacked state constraint vector.
    '''
    Fu_bar = np.kron(np.eye(N), Fu)
    fu_bar = np.kron(np.ones(N), fu)
    Fx_bar = np.kron(np.eye(N), Fx)
    fx_bar = np.kron(np.ones(N), fx)
    return Fu_bar, fu_bar, Fx_bar, fx_bar


def _lqr_gain(A, B, Q, R):
    '''
    Computes the discrete-time LQR gain K for state feedback u = -Kx.

    Parameters:
            A (np.ndarray): Discrete-time state matrix.
            B (np.ndarray): Discrete-time input matrix.
            Q (np.ndarray): State weighting matrix.
            R (float): Input weighting scalar.

    Returns:
            K (np.ndarray): The optimal LQR gain matrix.
    '''
    P = la.solve_discrete_are(A, B, Q, np.array([[R]]))
    return -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def _observer_gain(A_aug, C_aug):
    '''
    Computes the Luenberger observer gain L via pole placement.

    Parameters:
            A_aug (np.ndarray): Augmented state matrix.
            C_aug (np.ndarray): Augmented output matrix.

    Returns:
            L (np.ndarray): The observer gain matrix.
    '''
    n_aug = A_aug.shape[0]
    poles = np.linspace(0.7, 0.9, n_aug)
    L = signal.place_poles(A_aug.T, C_aug.T, poles).gain_matrix.T
    return L

def _remove_redundant_inequalities(F, f, tol=1e-9):
    '''
    Removes redundant inequality constraints from a polyhedron Fx <= f.

    Parameters:
            F (np.ndarray): Constraint matrix.
            f (np.ndarray): Constraint vector.
            tol (float): Numerical tolerance.

    Returns:
            F_reduced (np.ndarray): Reduced constraint matrix.
            f_reduced (np.ndarray): Reduced constraint vector.
    '''
    keep = []
    n_rows, n = F.shape

    for i in range(n_rows):
        c = -F[i]  # maximize F[i]x  <=>  minimize -F[i]x
        A_ub = np.delete(F, i, axis=0)
        b_ub = np.delete(f, i, axis=0)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * n, method="highs")

        if not res.success:
            # keep row if LP failed; conservative choice
            keep.append(i)
            continue

        max_val = F[i] @ res.x
        if max_val > f[i] + tol:
            keep.append(i) 

    return F[keep], f[keep]

def _same_polyhedron(F1, f1, F2, f2, tol=1e-9):
    '''
    Checks if two polyhedra defined by H-representation are identical.

    Parameters:
            F1 (np.ndarray): First constraint matrix.
            f1 (np.ndarray): First constraint vector.
            F2 (np.ndarray): Second constraint matrix.
            f2 (np.ndarray): Second constraint vector.
            tol (float): Numerical tolerance for comparison.

    Returns:
            (bool): True if the polyhedra are the same, False otherwise.
    '''
    if F1.shape != F2.shape or f1.shape != f2.shape:
        return False
    return np.allclose(F1, F2, atol=tol) and np.allclose(f1, f2, atol=tol)

def _terminal_set_matrices(A, B, Fx, fx, Fu, fu, Q, R, max_iter=200, tol=1e-9):
    '''
    Computes the maximal positively invariant terminal set.

    Parameters:
            A, B (np.ndarray): System matrices.
            Fx, fx (np.ndarray): State constraint polyhedron.
            Fu, fu (np.ndarray): Input constraint polyhedron.
            Q, R (np.ndarray, float): LQR weighting matrices.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.

    Returns:
            Ff (np.ndarray): Terminal set constraint matrix.
            ff (np.ndarray): Terminal set constraint vector.
    '''
    K = _lqr_gain(A, B, Q, R)
    A_K = A + B @ K

    # admissible set under u = Kx
    F_adm = np.vstack((Fx, Fu @ K))
    f_adm = np.concatenate((fx, fu))

    F_prev, f_prev = _remove_redundant_inequalities(F_adm, f_adm, tol=tol)

    # Iteratively compute the predecessor set and intersect with the admissible set
    for _ in range(max_iter):
        # predecessor under x+ = A_K x
        F_pre = F_prev @ A_K
        f_pre = f_prev.copy()

        # intersect admissible set with predecessor
        F_new = np.vstack((F_adm, F_pre))
        f_new = np.concatenate((f_adm, f_pre))

        F_new, f_new = _remove_redundant_inequalities(F_new, f_new, tol=tol)

        if _same_polyhedron(F_new, f_new, F_prev, f_prev, tol=tol):
            return F_new, f_new

        F_prev, f_prev = F_new, f_new

    # If max_iter is reached, return the best available approximation
    return F_prev, f_prev

def _check_mpc_feasible(x0, N, A, T, S, Fu_bar, fu_bar, Fx_bar, fx_bar, Ff, ff, H, Qbar):
    '''
    Checks if the MPC problem is feasible for a given initial state x0.

    Parameters:
            x0 (np.ndarray): Initial state.
            N (int): Prediction horizon.
            A, T, S (np.ndarray): System and prediction matrices.
            Fu_bar, fu_bar (np.ndarray): Stacked input constraints.
            Fx_bar, fx_bar (np.ndarray): Stacked state constraints.
            Ff, ff (np.ndarray): Terminal set constraints.
            H, Qbar (np.ndarray): QP cost matrices.

    Returns:
            (bool): True if the QP is feasible, False otherwise.
    '''
    n = A.shape[0]

    T_N = T[(N - 1) * n:N * n, :]
    S_N = S[(N - 1) * n:N * n, :]

    G = np.vstack((Fu_bar, Fx_bar @ S, Ff @ S_N))
    g = np.concatenate((fu_bar, fx_bar - Fx_bar @ T @ x0, ff - Ff @ T_N @ x0))
    h = 2 * (S.T @ Qbar @ T @ x0).flatten()

    U = solve_qp(H, h, G, g, solver="osqp")

    return U is not None

def _plot_study(t, data, titles, ylabels, constraint_lines=None, step_indices=None, ylimits=None):
    '''
    Generic plotting function for study results.

    Parameters:
            t (np.ndarray): Time vector.
            data (list): List of traces to plot.
            titles (list): List of subplot titles.
            ylabels (list): List of y-axis labels.
            constraint_lines (list, optional): Lines to draw for constraints.
            step_indices (set, optional): Indices of subplots to use step plot.
            ylimits (list, optional): List of y-axis limits.
    '''
    n_ax = len(titles)
    _, axs = plt.subplots(n_ax, 1, figsize=(10, 4*n_ax), sharex=True)
    if n_ax == 1: 
        axs = [axs]
    step_indices = step_indices or set()

    for i, (title, ylabel) in enumerate(zip(titles, ylabels)):
        axs[i].set_title(title, fontsize=11, fontweight='bold')
        axs[i].set_ylabel(ylabel, fontsize=11)
        axs[i].grid(True, alpha=0.4)

    for (ax_i, label, color, ls, lw, y) in data:
        if ax_i in step_indices:
            axs[ax_i].step(t[:-1], y, where='post', color=color,
                           linestyle=ls, linewidth=lw, label=label)
        else:
            axs[ax_i].plot(t, y, color=color, linestyle=ls,
                           linewidth=lw, label=label)

    for (ax_i, yval, color, ls, label) in (constraint_lines or []):
        axs[ax_i].axhline(yval, color=color, linestyle=ls, linewidth=1.2,
                          alpha=0.7, label=label)

    for ax in axs:
        ax.legend(loc='upper right', fontsize=9)
    axs[-1].set_xlabel('Time [s]', fontsize=11)

    if ylimits is not None:
        for ax, lim in zip(axs, ylimits):
            if lim is not None:
                ax.set_ylim(lim)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. SIMULATION FUNCTIONS
# =============================================================================

def simulate_MPC(x0, N, N_sim, Q, R):
    '''
    Simulates the closed-loop MPC controller.

    Parameters:
            x0 (np.ndarray): Initial state vector.
            N (int): Prediction horizon.
            N_sim (int): Number of simulation steps.
            Q (np.ndarray): State weighting matrix.
            R (float): Input weighting scalar.

    Returns:
            x_list (np.ndarray): State trajectory.
            u_list (np.ndarray): Input trajectory.
            t (np.ndarray): Time vector.
            comp_time (float): Total computation time.
    '''
    A, B, Ts = get_system_matrices()
    n, m = A.shape[0], B.shape[1]
    Fu, fu, Fx, fx = _constraints()
    T, S = _prediction_matrices(A, B, N)
    Fu_bar, fu_bar, Fx_bar, fx_bar = _stack_constraints(Fu, fu, Fx, fx, N)
    H, Qbar, P_dare = _qp_cost(A, B, Q, R, N)
    Ff, ff = _terminal_set_matrices(A, B, Fx, fx, Fu, fu, Q, R)

    T_N, S_N = T[(N-1)*n:N*n, :], S[(N-1)*n:N*n, :]

    x_list = np.zeros((n, N_sim+1))
    u_list = np.zeros((m, N_sim))
    xk = x0.copy()
    x_list[:, 0] = xk

    t_start = time.perf_counter()

    for k in range(N_sim):
        G = np.vstack((Fu_bar, Fx_bar @ S, Ff @ S_N))
        g = np.concatenate((fu_bar, fx_bar - Fx_bar @ T @ xk, ff - Ff @ T_N @ xk))
        h = 2*(S.T @ Qbar @ T @ xk).flatten()
        U = solve_qp(H, h, G, g, solver='osqp')
        uk = np.array([U[0]]) if U is not None else np.zeros(m)

        xk = A @ xk + B @ uk
        u_list[:, k] = uk
        x_list[:, k+1] = xk

    comp_time = time.perf_counter() - t_start

    return x_list, u_list, np.arange(N_sim+1)*Ts, comp_time


def simulate_LQR(x0, N_sim, Q, R):
    '''
    Simulates a saturated LQR controller for comparison.

    Parameters:
            x0 (np.ndarray): Initial state vector.
            N_sim (int): Number of simulation steps.
            Q (np.ndarray): LQR state weighting matrix.
            R (float): LQR input weighting scalar.

    Returns:
            x_list (np.ndarray): State trajectory.
            u_list (np.ndarray): Input trajectory.
            t (np.ndarray): Time vector.
    '''
    A, B, Ts = get_system_matrices()
    n, m     = A.shape[0], B.shape[1]
    K        = _lqr_gain(A, B, Q, R)
    x_list   = np.zeros((n, N_sim+1));  u_list = np.zeros((m, N_sim))
    xk = x0.copy();  x_list[:, 0] = xk
    for k in range(N_sim):
        uk = np.clip(K @ xk, -10., 10.)
        xk = A @ xk + B @ uk
        u_list[:, k] = uk
        x_list[:, k+1] = xk
    return x_list, u_list, np.arange(N_sim+1)*Ts

def simulate_MPC_disturbance(x0, N, N_sim, Q, R, d_val, noise_std=0.0):
    '''
    Simulates an offset-free MPC with a Luenberger observer.

    Parameters:
            x0 (np.ndarray): Initial state vector.
            N (int): Prediction horizon.
            N_sim (int): Number of simulation steps.
            Q (np.ndarray): State weighting matrix.
            R (float): Input weighting scalar.
            d_val (float): Magnitude of the constant disturbance.
            noise_std (float): Standard deviation of measurement noise.

    Returns:
            x_list (np.ndarray): State trajectory.
            u_list (np.ndarray): Input trajectory.
            t (np.ndarray): Time vector.
    '''
    # 1. System Setup
    A, B, Ts = get_system_matrices()
    n, m = A.shape[0], B.shape[1]
    nd = 2  # Two disturbance channels
    C = np.eye(n)  # Full state measurements

    # Disturbance Model: d[0] affects pendulum angle dynamics (x[1]),
    # and d[1] affects pendulum velocity dynamics (x[3]).
    d_true = d_val * np.array([1., 1.])
    Cd = np.zeros((n, nd))
    Cd[1, 0] = 1.0
    Cd[3, 1] = 1.0

    # 2. Augmented Observer Construction
    # x_aug = [x; d] -> (n+nd) dimensional
    A_aug = np.block([[A, np.zeros((n, nd))],
                      [np.zeros((nd, n)), np.eye(nd)]])
    B_aug = np.vstack([B, np.zeros((nd, m))])
    C_aug = np.hstack([C, Cd])

    # Calculate Luenberger gain L
    L = _observer_gain(A_aug, C_aug)

    # 3. MPC Prediction Matrices
    Fu, fu, Fx, fx = _constraints()
    T, S = _prediction_matrices(A, B, N)
    Fu_bar, fu_bar, Fx_bar, fx_bar = _stack_constraints(Fu, fu, Fx, fx, N)
    Ff, ff = _terminal_set_matrices(A, B, Fx, fx, Fu, fu, Q, R)
    T_N = T[(N - 1) * n:N * n, :]
    S_N = S[(N - 1) * n:N * n, :]
    G_nom = np.vstack((Fu_bar, Fx_bar @ S, Ff @ S_N))
    H, Qbar, _ = _qp_cost(A, B, Q, R, N)

    # 4. Target Selection Parameters
    Q_target = np.diag([1.0, 1.0, 1.0, 1.0])
    R_target = np.array([[0.1]])
    H_ots = np.block([[Q_target, np.zeros((n, m))],
                      [np.zeros((m, n)), R_target]])

    # Simulation Buffers
    x_list = np.zeros((n, N_sim + 1))
    u_list = np.zeros((m, N_sim))
    x_true = x0.copy()
    x_aug_hat = np.zeros(n + nd)
    x_list[:, 0] = x_true

    for k in range(N_sim):
        # --- A. Measurement & State Estimation ---
        y_k = C @ x_true + Cd @ d_true

        # Add Gaussian measurement noise if given
        if noise_std > 0.0:
            noise = np.random.normal(0.0, noise_std, size=n)
            y_k = y_k + noise

        # Update augmented state: [x_hat; d_hat]
        x_aug_hat = x_aug_hat + L @ (y_k - C_aug @ x_aug_hat)

        x_hat = x_aug_hat[:n]
        d_hat = x_aug_hat[n:]

        # --- B. Optimal Target Selection (OTS) ---
        A_eq = np.block([[np.eye(n) - A, -B],
                         [C, np.zeros((n, m))]])
        b_eq = np.concatenate([np.zeros(n), -(Cd @ d_hat)])

        A_iq = np.block([[np.zeros((fu.size, n)), Fu],
                         [Fx, np.zeros((fx.size, m))]])
        b_iq = np.concatenate([fu, fx])

        def target_obj(z):
            return 0.5 * z.T @ H_ots @ z

        # Optimization:
        res = minimize(target_obj, np.zeros(n + m),
                       constraints=[{'type': 'eq', 'fun': lambda z: A_eq @ z - b_eq},
                                    {'type': 'ineq', 'fun': lambda z: b_iq - A_iq @ z}])

        xr = res.x[:n]
        ur = res.x[n:]

        # --- C. Solve MPC QP for Delta-u ---
        e0 = x_hat - xr

        # Stack the steady-state target over the horizon
        xr_bar = np.tile(xr, N)
        ur_bar = np.tile(ur, N)

        # Adjust constraints for the target offset
        g_qp = np.concatenate((fu_bar - Fu_bar @ ur_bar, 
                               fx_bar - Fx_bar @ xr_bar - Fx_bar @ T @ e0,
                               ff - Ff @ T_N @ e0))
        h_lin = 2 * (S.T @ Qbar @ T @ e0).flatten()

        # Standard QP solve
        U = solve_qp(H, h_lin, G_nom, g_qp, solver='osqp')

        # Apply control and step simulation
        uk = np.clip((U[:m] + ur) if U is not None else ur, -10., 10.)

        x_true = A @ x_true + B @ uk
        x_aug_hat = A_aug @ x_aug_hat + B_aug @ uk

        u_list[:, k] = uk
        x_list[:, k + 1] = x_true

    return x_list, u_list, np.arange(N_sim + 1) * Ts

def check_augmented_system_rank(Cd, Bd):
    '''
    Checks the rank condition for the detectability of the augmented system.

    Parameters:
            Cd (np.ndarray): Disturbance measurement matrix.
            Bd (np.ndarray): Disturbance input matrix.
    '''
    # Retrieve system matrices to extract state matrix A
    A, _, _ = get_system_matrices()
    C = np.eye(4)
    n = A.shape[0]
    nd = Cd.shape[1]

    # Construct the augmented block matrix to verify detectability
    aug_M = np.block([[np.eye(n) - A, Bd],
                      [C, Cd]])
    nec_rank = n + nd

    print(f'For {nd} disturbance states:')
    print('Full augmented matrix:')
    print(aug_M)
    print(f'Rank: {np.linalg.matrix_rank(aug_M)}')
    print(f'n = {n}, n_d = {nd}, so necessary rank = {nec_rank}.')

    # Evaluate if the augmented matrix meets the necessary rank condition
    if np.linalg.matrix_rank(aug_M) >= nec_rank:
        print('Rank condition holds')
    else:
        print('Rank condition does not hold')
    print('\n')

def estimate_roa_2d(var_1, var_2, N, Q, R, idx_1, idx_2, title):
    '''
    Estimates and plots a 2D slice of the region of attraction (RoA).

    Parameters:
            var_1 (np.ndarray): Grid values for the first state variable.
            var_2 (np.ndarray): Grid values for the second state variable.
            N (int): Prediction horizon.
            Q (np.ndarray): MPC state weighting matrix.
            R (float): MPC input weighting scalar.
            idx_1 (int): Index of the first state variable.
            idx_2 (int): Index of the second state variable.
            title (str): Plot title.
    '''
    fixed_state = np.zeros(4)

    # System + constraints
    A, B, _ = get_system_matrices()
    Fu, fu, Fx, fx = _constraints()
    T, S = _prediction_matrices(A, B, N)
    Fu_bar, fu_bar, Fx_bar, fx_bar = _stack_constraints(Fu, fu, Fx, fx, N)
    H, Qbar, _ = _qp_cost(A, B, Q, R, N)

    # Reuse terminal set helper
    Ff, ff = _terminal_set_matrices(A, B, Fx, fx, Fu, fu, Q, R)

    # Numerical RoA estimation
    feasible = np.zeros((len(var_2), len(var_1)), dtype=int)

    for i, v2 in enumerate(var_2):
        for j, v1 in enumerate(var_1):
            x0 = fixed_state.copy()
            x0[idx_1] = v1
            x0[idx_2] = v2

            feasible[i, j] = int(_check_mpc_feasible(x0, N, A, T, S, Fu_bar, fu_bar, 
                                                     Fx_bar, fx_bar, Ff, ff, H, Qbar))

    # Optional terminal-set slice on same grid
    terminal_mask = np.zeros((len(var_2), len(var_1)), dtype=int)
    for i, v2 in enumerate(var_2):
        for j, v1 in enumerate(var_1):
            x = fixed_state.copy()
            x[idx_1] = v1
            x[idx_2] = v2
            terminal_mask[i, j] = int(np.all(Ff @ x <= ff + 1e-9))

    # Plot
    V1, V2 = np.meshgrid(var_1, var_2)

    # custom colormap: 0 = infeasible (red), 1 = feasible (green)
    cmap = ListedColormap(['red', 'green'])

    plt.figure(figsize=(8, 6))

    # filled plot
    plt.contourf(V1, V2, feasible,
                levels=[-0.5, 0.5, 1.5],
                cmap=cmap)

    # terminal set boundary
    plt.contour(V1, V2, terminal_mask,
                levels=[0.5],
                colors='k',
                linewidths=2)

    labels = [r'$\theta_a$', r'$\theta_p$', r'$\dot{\theta}_a$', r'$\dot{\theta}_p$']
    units  = ['[rad]', '[rad]', '[rad/s]', '[rad/s]']

    plt.xlabel(f'{labels[idx_1]} {units[idx_1]}')
    plt.ylabel(f'{labels[idx_2]} {units[idx_2]}')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=6, label=r'Region of Attraction $\mathcal{X}_N$'),
        Line2D([0], [0], color='red', lw=6, label='Infeasible'),
        Line2D([0], [0], color='black', lw=2, label=r'Terminal set $\mathbb{X}_f$')
    ]

    plt.legend(handles=legend_elements)

    plt.show()


# =============================================================================
# 5. STUDY FUNCTIONS
# =============================================================================

def discrete_time_matrices_study():
    """
    Runs and prints the study on system discretization and sampling time.
    """
    get_system_matrices(discrete_time_study=True)

def terminal_set_study():
    """
    Computes and prints the properties of the terminal set.
    """
    A, B, _ = get_system_matrices()
    Fu, fu, Fx, fx = _constraints()
    Q = np.diag([0.1, 10.0, 1.0, 1.0])
    R = 0.1

    F_prev, _ = _terminal_set_matrices(A, B, Fx, fx, Fu, fu, Q, R)
    print('\n')
    print('=========Terminal set study=========')
    print(f'Size of terminal set: {F_prev.shape}.')
    print('\n')

def region_of_attraction_study():
    '''
    Estimates and plots 2D slices of the Region of Attraction (RoA).
    '''
    N = 35

    # Slice: pendulum-angle / pendulum-speed slice
    var_1 = np.linspace(-0.4, 0.4, 50)  # theta_p
    var_2 = np.linspace(-8, 8, 50)      # omega_p

    estimate_roa_2d(
        var_1=var_1,
        var_2=var_2,
        N=N,
        Q=np.diag([0.1, 10., 1., 1.]),
        R=0.1,
        idx_1=1,
        idx_2=3,
        title='2D Region of Attraction for pendulum states'
    )

    # Slice: arm-speed / pendulum-speed slice
    var_1 = np.linspace(-8, 8, 50)      # theta_p
    var_2 = np.linspace(-8, 8, 50)      # omega_p

    estimate_roa_2d(
        var_1=var_1,
        var_2=var_2,
        N=N,
        Q=np.diag([0.1, 10., 1., 1.]),
        R=0.1,
        idx_1=2,
        idx_2=3,
        title='2D Region of Attraction for angular velocities'
    )

def horizon_study():
    '''
    Studies the effect of the prediction horizon N on performance.
    '''
    x0, Q, R, N_sim = np.array([0.,0.,0.,3.5]), np.diag([0.1, 10.0, 1.0, 1.0]), 0.1, 100

    results = {}
    for N in [5, 15, 35, 50]:
        print(f"Simulating: N={N} ...")
        x, u, t, comp_time = simulate_MPC(x0, N, N_sim, Q, R)
        results[N, comp_time] = (x, u, t)

    styles = [('-',  '#0072B2'), ('-', '#009E73'), ('-',  '#D55E00'), ('-',  "#8000D5")]
    data, t = [], None
    for ((N, comp_time), (x, u, tv)), (ls, col) in zip(results.items(), styles):
        t = tv
        data += [(0, f'N={N} | computation time = {comp_time:.2f}s', col, ls, 2., x[1,:])]

    _plot_study(t, data,
                titles=['Pendulum Angle for Horizon Study'],
                ylabels=[r'$\theta_p$ [rad]'],
                constraint_lines=[(0, 0.3,'r','--',r'Constraint $\pm$0.3'),
                                  (0,-0.3,'r','--','')],
                ylimits = [(-0.4, 0.4)])


def weight_tuning_study():
    '''
    Studies the effect of MPC weight matrices (Q and R) on performance.
    '''
    x0, N, N_sim = np.array([0.,0.,0.,3.5]), 35, 100
    configs = {
        "Non-uniform Q with cheap control":         (np.diag([0.1, 10.0, 1.0, 1.0]), 0.1),
        "Non-uniform Q with expensive control":     (np.diag([0.1, 10.0, 1.0, 1.0]), 10.0),
        "Uniform Q with cheap control":             (np.diag([1.0, 1.0, 1.0, 1.0]),  0.1),
        "Uniform Q with expensive control":         (np.diag([1.0, 1.0, 1.0, 1.0]),  10.0),
    }
    results = {}
    for name, (Q, R) in configs.items():
        print(f"Simulating: {name} ...")
        x, u, t, _ = simulate_MPC(x0, N, N_sim, Q, R)
        results[name] = (x, u, t)

    styles = [('-',  '#0072B2'), ('-', '#E69F00'), ('-', '#009E73'), ('-',  '#D55E00')]
    data, t = [], None
    for (name, (x, u, tv)), (ls, col) in zip(results.items(), styles):
        t = tv
        data += [(0, name, col, ls, 2., x[1,:]),
                 (1, '',   col, ls, 2., x[3,:]),
                 (2, '',   col, ls, 2., u[0,:])]

    _plot_study(t, data,
                titles=['Pendulum Angle', 'Pendulum Speed', 'Control Input'],
                ylabels=[r'$\theta_p$ [rad]', r'$\dot \theta_p$ [rad/s]', r'$u$ [V]'],
                constraint_lines=[(0, 0.3,'r','--',r'Constraint $\pm$0.3'),
                                  (0,-0.3,'r','--',''),
                                  (1, 2*np.pi,'r','--',r'Constraint $\pm 2\pi$'),
                                  (1,-2*np.pi,'r','--',''),
                                  (2, 10.,'r','--',r'Limit $\pm$10V'),
                                  (2,-10.,'r','--','')],
                step_indices={2})


def MPC_LQR_comparison_study():
    '''
    Compares the performance of MPC with a saturated LQR controller.
    '''
    x0  = np.array([0.,0.,0.,3.5])
    N, N_sim, Q, R = 35, 100, np.diag([0.1, 10.0, 1.0, 1.0]), 0.1
    x_lqr, u_lqr, t = simulate_LQR(x0, N_sim, Q, R)
    x_mpc, u_mpc, _, _ = simulate_MPC(x0, N, N_sim, Q, R)

    data = [(0,'Saturated LQR','b','--',2.,x_lqr[1,:]),
            (0,f'MPC','g','-', 2.,x_mpc[1,:]),
            (1,'','b','--',2.,x_lqr[3,:]),
            (1,'','g','-', 2.,x_mpc[3,:]),
            (2,'','b','--',2.,u_lqr[0,:]),
            (2,'','g','-', 2.,u_mpc[0,:])]

    _plot_study(t, data,
                titles=['Pendulum Angle','Pendulum Speed','Control Input'],
                ylabels=[r'$\theta_p$ [rad]',r'$\dot \theta_p$ [rad/s]',r'$u$ [V]'],
                constraint_lines=[(0, 0.3,'r',':',r'Constraint $\pm$0.3'),
                                  (0,-0.3,'r',':',''),
                                  (1, 2*np.pi,'r','--',r'Constraint $\pm 2\pi$'),
                                  (1,-2*np.pi,'r','--',''),
                                  (2, 10.,'r',':',r'Limit $\pm$10V'),
                                  (2,-10.,'r',':','')],
                step_indices={2})

def detectability_study_augmented_system():
    '''
    Evaluates and compares the detectability of the augmented system 
    under different disturbance models (4 vs. 2 disturbance states).
    '''
    # Model 1: Disturbance affects all 4 states
    Cd4 = np.eye(4)
    Bd4 = np.zeros((4, 4))

    # Model 2: Disturbance only affects the pendulum angle and velocity
    Cd2 = np.zeros((4, 2))
    Cd2[1, 0] = 1.0
    Cd2[3, 1] = 1.0
    Bd2 = np.zeros((4, 2))

    print('\n')
    print('=========Detectability of augmented system study=========')
    
    # Check rank condition for the 4-state disturbance model
    check_augmented_system_rank(Cd4, Bd4)
    
    # Check rank condition for the 2-state disturbance model
    check_augmented_system_rank(Cd2, Bd2)

def disturbance_noise_rejection_study():
    '''
    Evaluates offset-free MPC performance with disturbances and noise.
    '''
    x0 = np.array([0., 0., 0., 0.])
    N, N_sim, Q, R = 35, 150, np.diag([0.1, 10., 1., 1.]), 0.1
    d_val = 0.1

    Ts = get_system_matrices()[2]
    t_list = np.arange(N_sim + 1) * Ts

    print("Simulating Offset-free MPC (Disturbed, Clean)...")
    x_mpc, u_mpc, _ = simulate_MPC_disturbance(x0, N, N_sim, Q, R, d_val)

    print("Simulating Offset-free MPC (Disturbed + Noisy)...")
    x_mpc_n, u_mpc_n, _ = simulate_MPC_disturbance(x0, N, N_sim, Q, R, d_val, noise_std=0.01)

    data = [
        # Pendulum Angle
        (0, 'MPC (Clean)', 'g', '-', 1.2, x_mpc[1, :]),
        (0, 'MPC (Noisy)', 'r', '-', 1.2, x_mpc_n[1, :]),

        # Arm Angle
        (1, '', 'g', '-', 1.2, x_mpc[3, :]),
        (1, '', 'r', '-', 1.2, x_mpc_n[3, :]),

        # Control Input
        (2, '', 'g', '-', 1.2, u_mpc[0, :]),
        (2, '', 'r', '-', 1.2, u_mpc_n[0, :])
    ]

    _plot_study(t_list, data,
                titles=['Pendulum Angle',
                        'Pendulum Speed',
                        'Control Input'],
                ylabels=[r'$\theta_p$ [rad]', r'$\dot \theta_p$ [rad/s]', r'$u$ [V]'],
                constraint_lines=[(0, 0.0, 'k', '-', ''),
                                  (1, 0.0, 'k', '-', ''),
                                  (2, 10.0, 'r', '--', r'Limit $\pm$10V'),
                                  (2, -10.0, 'r', '--', '')],
                step_indices={2})


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Uncomment the desired study function to run it.

    discrete_time_matrices_study()
    # terminal_set_study()
    # region_of_attraction_study()
    # horizon_study()
    # weight_tuning_study()
    # MPC_LQR_comparison_study()
    # detectability_study_augmented_system()
    # disturbance_noise_rejection_study()
