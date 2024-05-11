import pulp as lpp
import numpy as np

w = [10, 1, 100]
c1 = [1, 1, 100, 100, 100, 100]
c2 = [1, 1, 100, 100, 100, 100]
c3 = [1, 1, 100, 100, 100, 100]

def transform_product(F_t, p, t):
    product = [ p[t, 0] + F_t[2, 0] * p[t, 2] + F_t[2, 1] * p[t, 3],
                p[t, 1] + F_t[2, 0] * p[t, 4] + F_t[2, 1] * p[t, 5],
                F_t[0, 0] * p[t, 2] + F_t[0, 1] * p[t, 3],
                F_t[1, 0] * p[t, 2] + F_t[1, 1] * p[t, 3],
                F_t[0, 0] * p[t, 4] + F_t[0, 1] * p[t, 5],
                F_t[1, 0] * p[t, 4] + F_t[1, 1] * p[t, 5]
            ]
    return product

def stabilize(F_t, shape, first_window=True, prev_Bt=None,ratio=0.8):
    # Create lpp minimization problem object
    problem = lpp.LpProblem("stabilize", lpp.LpMinimize)
    n_frames = len(F_t)
    crop_w = round(shape[1] * ratio)
    crop_h = round(shape[0] * ratio)
    crop_x = round(round(shape[1] / 2) - crop_w / 2)
    crop_y = round(round(shape[0] / 2) - crop_h / 2)
    corner_points = [
        (crop_x, crop_y),
        (crop_x + crop_w, crop_y),
        (crop_x, crop_y + crop_h),
        (crop_x + crop_w, crop_y + crop_h)
    ]
    e1 = lpp.LpVariable.dicts("e1", ((i, j) for i in range(n_frames) for j in range(6)), lowBound=0.0)
    e2 = lpp.LpVariable.dicts("e2", ((i, j) for i in range(n_frames) for j in range(6)), lowBound=0.0)
    e3 = lpp.LpVariable.dicts("e3", ((i, j) for i in range(n_frames) for j in range(6)), lowBound=0.0)
    p = lpp.LpVariable.dicts("p", ((i, j) for i in range(n_frames) for j in range(6)))
    # Construct objective to be minimized using e1, e2 and e3
    problem += w[0] * lpp.lpSum([e1[i, j] * c1[j] for i in range(n_frames) for j in range(6)]) + \
            w[1] * lpp.lpSum([e2[i, j] * c2[j] for i in range(n_frames) for j in range(6)]) + \
            w[2] * lpp.lpSum([e3[i, j] * c3[j] for i in range(n_frames) for j in range(6)])
    # Apply smoothness constraints on the slack variables e1, e2 and e3 using params p
    for t in range(n_frames - 3):
        res_t_prod = transform_product(F_t[t + 1], p, t + 1)
        res_t1_prod = transform_product(F_t[t + 2], p, t + 2)
        res_t2_prod = transform_product(F_t[t + 3], p, t + 3)
        res_t = [res_t_prod[j] - p[t, j] for j in range(6)]
        res_t1 = [res_t1_prod[j] - p[t + 1, j] for j in range(6)]
        res_t2 = [res_t2_prod[j] - p[t + 2, j] for j in range(6)]
        for j in range(6):
            problem += -1*e1[t, j] <= res_t[j]
            problem += e1[t, j] >= res_t[j]
            problem += -1 * e2[t, j] <= res_t1[j] - res_t[j]
            problem += e2[t, j] >= res_t1[j] - res_t[j]
            problem += -1 * e3[t, j] <= res_t2[j] - 2*res_t1[j] + res_t[j]
            problem += e3[t, j] >= res_t2[j] - 2*res_t1[j] + res_t[j]
    # Constraints
    for t1 in range(n_frames):
        # Proximity Constraints
        # For a_t
        problem += p[t1, 2] >= 0.9
        problem += p[t1, 2] <= 1.1
        # For b_t
        problem += p[t1, 3] >= -0.1
        problem += p[t1, 3] <= 0.1
        # For c_t
        problem += p[t1, 4] >= -0.1
        problem += p[t1, 4] <= 0.1
        # For d_t
        problem += p[t1, 5] >= 0.9
        problem += p[t1, 5] <= 1.1
        # For b_t + c_t
        problem += p[t1, 3] + p[t1, 4] >= -0.1
        problem += p[t1, 3] + p[t1, 4] <= 0.1
        # For a_t - d_t
        problem += p[t1, 2] - p[t1, 5] >= -0.05
        problem += p[t1, 2] - p[t1, 5] <= 0.05
        # Inclusion Constraints
        for (cx, cy) in corner_points:
            problem += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy >= 0
            problem += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy <= shape[1]
            problem += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy >= 0
            problem += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy <= shape[0]
    # Continuity constraints
    if not first_window:
        problem += p[0, 0] == prev_Bt[2, 0]
        problem += p[0, 1] == prev_Bt[2, 1]
        problem += p[0, 2] == prev_Bt[0, 0]
        problem += p[0, 3] == prev_Bt[1, 0]
        problem += p[0, 4] == prev_Bt[0, 1]
        problem += p[0, 5] == prev_Bt[1, 1]

    # Saliency Constraints
    #landmarks_array = get_landmarks_array(in_file)
    #print(landmarks_array)

    problem.writeLP("formulation.lp")
    problem.solve()

    B_t = np.zeros((n_frames, 3, 3), np.float32)
    B_t[:, :, :] = np.eye(3)
    if problem.status == 1:
        print("Solution converged")
        for i in range(n_frames):
            B_t[i, :, :2] = np.array([[p[i, 2].varValue, p[i, 4].varValue],
                                      [p[i, 3].varValue, p[i, 5].varValue],
                                      [p[i, 0].varValue, p[i, 1].varValue]])
    else:
        print("Error: Linear Programming problem status:", lpp.LpStatus[problem.status])
    return B_t